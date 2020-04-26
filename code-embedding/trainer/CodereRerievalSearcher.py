import torch
from torch import zeros
from torch.autograd import Variable
from torch import cosine_similarity
from utils.util import write_pkl, load_pkl, filter_digit_english, get_tokens, get_stemmed_words

class CodereRerievalSearcher():
    def __init__(self, candidate_num, code_body, func_name, all_dict, opt, model, data_loader=None):
        self.opt = opt
        self.func_name = func_name
        self.model = model
        self.data_loader = data_loader
        self.candidate_num = candidate_num
        self.candidate = list()
        self.candidate_index = list()
        self.candidate_sim = list()
        self.candidate_func = list()
        self.code_body = code_body
        self.comment_dict = all_dict.description
        self.vec_database = list()
        self.query = '<blank>'
        self.query_tokens = ['<blnak>']

    def search(self, query, to_disc=True, from_disc=False):
        self.query = query
        query_data = self.preprocess(query)
        query_tensor = torch.tensor(query_data).view(1, -1)
        self.model.eval()
        self.model.code_encoder.eval()
        self.model.comment_encoder.eval()
        with torch.no_grad():
            hidden = (Variable(zeros(2 * self.opt.nlayers, 1, self.opt.nhid)),
                      Variable(zeros(2 * self.opt.nlayers, 1, self.opt.nhid)))
            query_vec = self.model.comment_encoder(query_tensor, hidden)
            if not from_disc:
                for data_batch in self.data_loader:
                    meth_data, token_data, api_data, comment_data = data_batch
                    code_data = (meth_data, token_data, api_data)
                    code_vects =  self.model.code_encoder(code_data)
                    for code_vect in code_vects:
                        self.vec_database.append(code_vect.view(1, -1))
            else:
                self.vec_database = self.load('./data/java/temp/database_vectors.pkl')
            zero_data = torch.zeros((len(self.vec_database), self.opt.nout)).float()
            query_vecs = query_vec + zero_data
            database_vecs = torch.cat(self.vec_database, 0)
            result = cosine_similarity(query_vecs, database_vecs)
            self.candidate_sim = result.tolist()
            self.candidate_index = result.argsort()[:self.candidate_num].tolist()
            self.candidate = [self.code_body[i] for i in self.candidate_index]
            self.candidate_func = [self.func_name[i] for i in self.candidate_index]

        if to_disc:
            self.save(self.vec_database)
        return self.candidate_sim, self.candidate

    def save(self, vecs):
        write_pkl(vecs, './data/java/temp/database_vectors.pkl')

    def load(self, path):
        return load_pkl(path)

    def preprocess(self, text):
        query = text.replace("How to", "").replace("How do I", ""). \
            replace("How do you", "").replace("How do we", ""). \
            replace("How can I", "").replace("How can we", ""). \
            replace("What is", "").replace("How are", ""). \
            replace("How is", "").replace("What are", ""). \
            replace("Can I", "").replace("Can you", ""). \
            replace("Can we", ""). \
            replace("how to", "").replace("how do I", ""). \
            replace("how do you", "").replace("how do we", ""). \
            replace("how can I", "").replace("how can we", ""). \
            replace("what is", "").replace("how are", ""). \
            replace("how is", "").replace("what are", ""). \
            replace("can I", "").replace("can you", ""). \
            replace("can we", "")

        query = filter_digit_english(query)
        query_list = get_tokens(query)
        self.query_tokens = get_stemmed_words(query_list)
        return self.comment_dict.convertToIdx( self.query_tokens, 30)

    def save_result(self):
        with open('./result.txt', 'w+') as f:
            head = [f'{self.query}\n' \
                    f'#################\n' \
                    f'query:{self.query_tokens}\n' \
                    f'#################\n\n']
            result = [f'#{i+1} sim: {self.candidate_sim[i]}\n' \
                      f'---\n' \
                      f'{self.candidate_func[i]}\n' \
                      f'---\n' \
                      f'{self.candidate[i]}\n\n'
                      for i in range(self.candidate_num)]
            result_combine = head + result
            f.writelines(result_combine)
