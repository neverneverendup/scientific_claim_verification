import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.encode import encode_paragraph
from embedding.jointmodel import JointModelClassifier, AbstractRationaleJointModelClassifier
# from embedding.model import JointModelClassifier
from utils import token_idx_by_sentence, remove_dummy


def get_predictions(args, input_set, checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.batch_size_gpu = 8
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args).to(device)
    # model = JointParagraphClassifier(args.model, args.hidden_dim, args.dropout).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    # for m in model.state_dict().keys():
    #     print(m)
    # p
    abstract_result = []
    rationale_result = []
    retrieval_result = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(input_set, batch_size=10, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            # match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
            #                                       args.model, match=True)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            # match_indices = [tensor.to(device) for tensor in match_indices]
            # abstract_out, rationale_out = model(encoded_dict, transformation_indices, match_indices)
            abstract_out, rationale_out, retrieval_out = model(encoded_dict, transformation_indices)
            abstract_result.extend(abstract_out)
            rationale_result.extend(rationale_out)
            retrieval_result.extend(retrieval_out)

    return abstract_result, rationale_result, retrieval_result

def get_abstract_rationale_predictions(args, input_set, checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.batch_size_gpu = 8
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AbstractRationaleJointModelClassifier(args).to(device)
    # model = JointParagraphClassifier(args.model, args.hidden_dim, args.dropout).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    abstract_result = []
    rationale_result = []
    retrieval_result = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(input_set, batch_size=10, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            # encoded = encode_paragraph(tokenizer, batch['claim'], batch['paragraph'])
            # encoded = {key: tensor.to(device) for key, tensor in encoded.items()}
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            # match_indices = token_idx_by_sentence(encoded_dict["input_ids"], tokenizer.sep_token_id,
            #                                       args.model, match=True)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            # match_indices = [tensor.to(device) for tensor in match_indices]
            # abstract_out, rationale_out = model(encoded_dict, transformation_indices, match_indices)
            rationale_out, retrieval_out = model(encoded_dict, transformation_indices)
            #abstract_result.extend(abstract_out)
            rationale_result.extend(rationale_out)
            retrieval_result.extend(retrieval_out)

    return rationale_result, retrieval_result

def get_abstract_retrieval(args, input_set, checkpoint):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = JointModelClassifier(args).to(device)
    model.load_state_dict(torch.load(checkpoint))
    model.eval()
    retrieval_result = []
    with torch.no_grad():
        for batch in tqdm(DataLoader(input_set, batch_size=1, shuffle=False)):
            encoded_dict = encode_paragraph(tokenizer, batch['claim'], batch['abstract'])
            transformation_indices = token_idx_by_sentence(encoded_dict['input_ids'], tokenizer.sep_token_id,
                                                           args.model)
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            transformation_indices = [tensor.to(device) for tensor in transformation_indices]
            retrieval_out, _ = model(encoded_dict, transformation_indices,
                                     retrieval_only=True)
            retrieval_result.extend(retrieval_out)

    return retrieval_result
