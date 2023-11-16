import os, random
from utils.simple_solution import SimpleSolution
from utils.evaluator import Evaluator
from utils.data import Data
from ontolearn.knowledge_base import KnowledgeBase
from owlapy.render import DLSyntaxObjectRenderer
from roces import BaseConceptSynthesis
from roces.synthesizer import ConceptSynthesizer
from owlapy.parser import DLSyntaxParser
from dataloader import NCESDataLoaderInference2
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
import json
import torch
import numpy as np, time
from collections import defaultdict
import re


def build_roces_vocabulary(data_train, data_test, kb, args):
    def add_data_values(path):
        print("\n*** Finding relevant data values ***")
        values = set()
        for ce, lp in data_train+data_test:
            if '[' in ce:
                for val in re.findall("\[(.*?)\]", ce):
                    values.add(val.split(' ')[-1])
        print("*** Done! ***\n")
        print("Added values: ", values)
        print()
        return list(values)
    renderer = DLSyntaxObjectRenderer()
    individuals = [ind.get_iri().as_str().split("/")[-1] for ind in kb.individuals()]
    atomic_concepts = list(kb.ontology().classes_in_signature())
    atomic_concept_names = [renderer.render(a) for a in atomic_concepts]
    role_names = [rel.get_iri().get_remainder() for rel in kb.ontology().object_properties_in_signature()] + \
                 [rel.get_iri().get_remainder() for rel in kb.ontology().data_properties_in_signature()]
    vocab = atomic_concept_names + role_names + ['⊔', '⊓', '∃', '∀', '¬', '⊤', '⊥', '.', ' ', '(', ')',\
                                                '⁻', '≤', '≥', 'True', 'False', '{', '}', ':', '[', ']',
                                                'double', 'integer', 'date', 'xsd']
    quantified_restriction_values = [str(i) for i in range(1,12)]
    data_values = add_data_values(args.knowledge_base_path)
    vocab = vocab + data_values + quantified_restriction_values
    vocab = sorted(set(vocab)) + ['PAD']
    print("Vocabulary size: ", len(vocab))
    num_examples = min(args.num_examples, kb.individuals_count()//2)
    return vocab, num_examples

def before_pad(arg):
    arg_temp = []
    for atm in arg:
        if atm == 'PAD':
            break
        arg_temp.append(atm)
    return arg_temp

def compute_accuracy(prediction, target):
    def soft(arg1, arg2):
        arg1_ = arg1
        arg2_ = arg2
        if isinstance(arg1_, str):
            arg1_ = set(before_pad(BaseConceptSynthesis.decompose(arg1_)))
        else:
            arg1_ = set(before_pad(arg1_))
        if isinstance(arg2_, str):
            arg2_ = set(before_pad(BaseConceptSynthesis.decompose(arg2_)))
        else:
            arg2_ = set(before_pad(arg2_))
        return 100*float(len(arg1_.intersection(arg2_)))/len(arg1_.union(arg2_))

    def hard(arg1, arg2):
        arg1_ = arg1
        arg2_ = arg2
        if isinstance(arg1_, str):
            arg1_ = before_pad(BaseConceptSynthesis.decompose(arg1_))
        else:
            arg1_ = before_pad(arg1_)
        if isinstance(arg2_, str):
            arg2_ = before_pad(BaseConceptSynthesis.decompose(arg2_))
        else:
            arg2_ = before_pad(arg2_)
        return 100*float(sum(map(lambda x,y: x==y, arg1_, arg2_)))/max(len(arg1_), len(arg2_))
    soft_acc = sum(map(soft, prediction, target))/len(target)
    hard_acc = sum(map(hard, prediction, target))/len(target)
    return soft_acc, hard_acc

num_examples = 1000
def collate_batch(batch):
    pos_emb_list = []
    neg_emb_list = []
    target_labels = []
    for pos_emb, neg_emb, label in batch:
        if pos_emb.ndim != 2:
            pos_emb = pos_emb.reshape(1, -1)
        if neg_emb.ndim != 2:
            neg_emb = neg_emb.reshape(1, -1)
        pos_emb_list.append(pos_emb)
        neg_emb_list.append(neg_emb)
        target_labels.append(label)
    pos_emb_list[0] = F.pad(pos_emb_list[0], (0, 0, 0, num_examples - pos_emb_list[0].shape[0]), "constant", 0)
    pos_emb_list = pad_sequence(pos_emb_list, batch_first=True, padding_value=0)
    neg_emb_list[0] = F.pad(neg_emb_list[0], (0, 0, 0, num_examples - neg_emb_list[0].shape[0]), "constant", 0)
    neg_emb_list = pad_sequence(neg_emb_list, batch_first=True, padding_value=0)
    target_labels = pad_sequence(target_labels, batch_first=True, padding_value=-100)
    return pos_emb_list, neg_emb_list, target_labels

def predict(kb, test_data, models, embedding_models, repeat_pred, args):
    args.path_to_triples = f"datasets/{kb}/Triples/"
    global num_examples
    num_examples = models[0].num_examples
    vocab = models[0].vocab
    inv_vocab = models[0].inv_vocab
    kb_embedding_data = Data(args)
    soft_acc, hard_acc = 0.0, 0.0
    preds = []
    targets = []
    if repeat_pred:
        k_values = np.linspace(1+0.2*num_examples, 0.95*num_examples, num=5).astype(int)
        Scores = None
        print("k values:", k_values)
        for j,k in tqdm(enumerate(k_values), total=len(k_values), desc='sampling examples...'):
            test_dataset = NCESDataLoaderInference2(test_data, kb_embedding_data, k, vocab, inv_vocab, args, random_sample=True)
            for i, (model, embedding_model) in enumerate(zip(models, embedding_models)):
                model = model.eval()
                scores = []
                test_dataset.load_embeddings(embedding_model.eval())
                test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
                for x1, x2, labels in tqdm(test_dataloader):
                    if i == 0 and j == 0:
                        target_sequence = model.inv_vocab[labels]
                        targets.append(target_sequence) # The target sequence does not depend on the current model
                    _, sc = model(x1, x2)
                    scores.append(sc.detach()) 
                scores = torch.cat(scores, 0)
                if i == 0:
                    all_scores = scores
                else:
                    all_scores = all_scores + scores
            all_scores = all_scores / len(models)
            if j == 0:
                Scores = all_scores
            else:
                Scores = Scores + all_scores
        Scores = Scores / len(k_values)
    else:
        test_dataset = NCESDataLoaderInference2(test_data, kb_embedding_data, num_examples, vocab, inv_vocab, args)
        for i, (model, embedding_model) in enumerate(zip(models, embedding_models)):
            model = model.eval()
            scores = []
            test_dataset.load_embeddings(embedding_model.eval())
            test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_batch, shuffle=False)
            for x1, x2, labels in tqdm(test_dataloader):
                if i == 0:
                    target_sequence = model.inv_vocab[labels]
                    targets.append(target_sequence) # The target sequence does not depend on the current model
                _, sc = model(x1, x2)
                scores.append(sc.detach()) 
            scores = torch.cat(scores, 0)
            if i == 0:
                Scores = scores
            else:
                Scores = Scores + scores
        Scores = Scores / len(models)
            
    pred_sequence = model.inv_vocab[Scores.argmax(1)]
    targets = np.concatenate(targets, 0)
    assert len(targets) == len(pred_sequence), f"Something went wrong: len(targets) is {len(targets)} and len(predictions) is {len(pred_sequence)}"
    soft_acc, hard_acc = compute_accuracy(pred_sequence, targets)
    print(f"Average syntactic accuracy, Soft: {soft_acc}%, Hard: {hard_acc}%")
    return pred_sequence, targets

def initialize_synthesizer(vocab, num_examples, num_inds, args):
    args.num_inds = num_inds
    roces = ConceptSynthesizer(vocab, num_examples, args)
    roces.refresh()
    return roces.model, roces.embedding_model

def synthesize_class_expressions(kb, test_data, vocab, num_examples, num_inds, repeat_pred, args):
    args.knowledge_base_path = "datasets/"+f"{kb}/{kb}.owl"
    embs = torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points32.pt", map_location = torch.device("cpu"))
    setattr(args, 'num_entities', embs['emb_ent_real.weight'].shape[0])
    setattr(args, 'num_relations', embs['emb_rel_real.weight'].shape[0])
    models, embedding_models = [], []
    for inds in num_inds:
        model, embedding_model = initialize_synthesizer(vocab, num_examples, inds, args)
        if args.sampling_strategy != 'uniform':
            model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_inducing_points{inds}.pt",
                             map_location=torch.device("cpu")))
            embedding_model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_inducing_points{inds}.pt",
                                        map_location = torch.device("cpu")))
        
        else:
            model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/{args.kb_emb_model}_SetTransformer_uniform_inducing_points{inds}.pt",
                         map_location=torch.device("cpu")))
            embedding_model.load_state_dict(torch.load(f"datasets/{kb}/Model_weights/SetTransformer_{args.kb_emb_model}_Emb_uniform_inducing_points{inds}.pt",
                                    map_location = torch.device("cpu")))
        models.append(model); embedding_models.append(embedding_model)
    return predict(kb, test_data, models, embedding_models, repeat_pred, args)
    
                
def evaluate_ensemble(kb_name, args, repeat_pred=False, save_results=False, verbose=False):
    print('#'*50)
    print('ROCES evaluation on {} KB:'.format(kb_name))
    print('#'*50)
    all_metrics = {'+'.join(combine): defaultdict(lambda: defaultdict(list)) for combine in [["SetTransformer_I32", "SetTransformer_I64"], \
                                        ["SetTransformer_I32", "SetTransformer_I128"], ["SetTransformer_I64", "SetTransformer_I128"],\
                                        ["SetTransformer_I32", "SetTransformer_I64", "SetTransformer_I128"]]}
    print()
    kb = KnowledgeBase(path=f"datasets/{kb_name}/{kb_name}.owl")
    with open(f"datasets/{kb_name}/Test_data/Data.json", "r") as file:
        test_data = json.load(file)
    with open(f"datasets/{kb_name}/Train_data/Data.json", "r") as file:
        train_data = json.load(file)
    vocab, num_examples = build_roces_vocabulary(train_data, test_data, kb, args)
    namespace = list(kb.individuals())[0].get_iri().get_namespace()
    print("KB namespace: ", namespace)
    print()
    simpleSolution = SimpleSolution(kb)
    evaluator = Evaluator(kb)
    dl_parser = DLSyntaxParser(namespace = namespace)
    all_individuals = set(kb.individuals())
    for combine in all_metrics.keys():     
        t0 = time.time()
        num_inds = [int(model_name.split("I")[-1]) for model_name in combine.split("+")]
        predictions, targets = synthesize_class_expressions(kb_name, test_data, vocab, num_examples, num_inds, repeat_pred, args)
        t1 = time.time()
        duration = (t1-t0)/len(predictions)
        for i, pb_str in enumerate(targets):
            pb_str = "".join(before_pad(pb_str))
            try:
                end_idx = np.where(predictions[i] == 'PAD')[0][0] # remove padding token
            except IndexError:
                end_idx = -1
            pred = predictions[i][:end_idx]
            try:
                prediction = dl_parser.parse("".join(pred.tolist()))
            except Exception:
                try:
                    pred = simpleSolution.predict(predictions[i].sum())
                    prediction = dl_parser.parse(pred)
                except Exception:
                    print(f"Could not understand expression {pred}")
            if prediction is None:
                prediction = dl_parser.parse('⊤')
            target_expression = dl_parser.parse(pb_str) # The target class expression
            positive_examples = set(kb.individuals(target_expression))
            negative_examples = all_individuals-positive_examples
            try:
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            except Exception as err:
                print("Parsing error on ", prediction)
                print(err)
                prediction = dl_parser.parse('⊤')
                acc, f1 = evaluator.evaluate(prediction, positive_examples, negative_examples)
            if verbose:
                print(f'Problem {i}, Target: {pb_str}, Prediction: {simpleSolution.renderer.render(prediction)}, Acc: {acc}, F1: {f1}')
                print()
            all_metrics[combine]['acc']['values'].append(acc)
            try:
                all_metrics[combine]['prediction']['values'].append(simpleSolution.renderer.render(prediction))
            except:
                all_metrics[combine]['prediction']['values'].append("Unknown")
            all_metrics[combine]['f1']['values'].append(f1)
            all_metrics[combine]['time']['values'].append(duration)

        for metric in all_metrics[combine]:
            if metric != 'prediction':
                all_metrics[combine][metric]['mean'] = [np.mean(all_metrics[combine][metric]['values'])]
                all_metrics[combine][metric]['std'] = [np.std(all_metrics[combine][metric]['values'])]

        print(combine+' Speed: {}s +- {} / lp'.format(round(all_metrics[combine]['time']['mean'][0], 2),\
                                                               round(all_metrics[combine]['time']['std'][0], 2)))
        print(combine+' Avg Acc: {}% +- {} / lp'.format(round(all_metrics[combine]['acc']['mean'][0], 2),\
                                                               round(all_metrics[combine]['acc']['std'][0], 2)))
        print(combine+' Avg F1: {}% +- {} / lp'.format(round(all_metrics[combine]['f1']['mean'][0], 2),\
                                                               round(all_metrics[combine]['f1']['std'][0], 2)))

        print()

    if save_results:
        if args.sampling_strategy != 'uniform':
            if repeat_pred:
                with open(f"datasets/{kb_name}/Results/ROCES+_{args.kb_emb_model}_Ensemble.json", "w") as file:
                    json.dump(all_metrics, file, indent=3, ensure_ascii=False)
            else:
                with open(f"datasets/{kb_name}/Results/ROCES_{args.kb_emb_model}_Ensemble.json", "w") as file:
                    json.dump(all_metrics, file, indent=3, ensure_ascii=False)
        else:
            if repeat_pred:
                    with open(f"datasets/{kb_name}/Results/ROCES+_{args.kb_emb_model}_uniform_Ensemble.json", "w") as file:
                        json.dump(all_metrics, file, indent=3, ensure_ascii=False)
            else:
                with open(f"datasets/{kb_name}/Results/ROCES_{args.kb_emb_model}_uniform_Ensemble.json", "w") as file:
                    json.dump(all_metrics, file, indent=3, ensure_ascii=False)
