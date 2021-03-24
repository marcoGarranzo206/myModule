from transformers import BertTokenizer, BertModel
import torch

class tokenizeBert:

    def __init__(self, bert_type,tokenize_strategy):

        self.tokenizer = BertTokenizer.from_pretrained(bert_type)
        self.tokenize_strat = tokenize_strategy

        self.model = BertModel.from_pretrained(bert_type,
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        self.model.eval()

    def tokenize(self,text):

        """
        tokenize text using bert language model
        """

        marked = "[CLS] " + text + " [SEP]"
        tokenized = self.tokenizer.tokenize(marked)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized)
        segments_ids = [1] * len(tokenized)


        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        return tokenized, self._ret_tensor(tokens_tensor, segments_tensors)

    def _ret_tensor(self, tokens_tensor, segments_tensors):


        with torch.no_grad():

            outputs = self.model(tokens_tensor, segments_tensors)

            # Evaluating the model will return a different number of objects based on
            # how it's  configured in the `from_pretrained` call earlier. In this case,
            # becase we set `output_hidden_states = True`, the third item will be the
            # hidden states from all layers. See the documentation for more details:
            # https://huggingface.co/transformers/model_doc/bert.html#bertmodel
            hidden_states = outputs[2]


        token_embeddings = torch.stack(hidden_states, dim = 0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        if self.tokenize_strat == "last":

            return token_embeddings[:,-1,:]

        elif self.tokenize_strat == "first":

            return token_embeddings[:,0,:]

        elif self.tokenize_strat == "sum_all":

            return token_embeddings[:,1:,:].sum(dim = 1)

        elif self.tokenize_strat == "second_to_last":

            return token_embeddings[:,-2,:]

        elif self.tokenize_strat == "sum_last_four":

            return token_embeddings[:,-4:,:].sum(dim = 1)

        elif self.tokenize_strat == "concat":

            return token_embeddings[:,-4:,:].reshape( token_embeddings.shape[0],1,-1)

class tokenizer_seq2seq(tokenizeBert):
    
    def __init__(self, bert_type,tokenize_strategy):
        
        super(tokenizer_seq2seq, self).__init__(bert_type, tokenize_strategy)
        
    def tokenize(self, text, outputs):
        
        """
        tokenize text using bert language model for seq2seq tasks
        
        text has to be a list of tokens of size L
        output is a list of outputs (POS, NER etc...) of size L
        
        bert can split words into separate tokens, so the output for these words
        also has to be split
        
        ie word embeddings has POS X
        split into 3 tokens, each which retains the POS X 
        
        em--------X
        ##bed-----X
        ##ings----X
        """
        
        tokenized_text = []
        extended_outputs = []
        for word, o in zip(text, outputs):
            
            tokenized = self.tokenizer.tokenize(word)
            n_subwords = len(tokenized)
            extended_outputs.extend([o]*n_subwords)
            tokenized_text.extend(tokenized)
            
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        return tokenized_text, self._ret_tensor(tokens_tensor, segments_tensors), extended_outputs

