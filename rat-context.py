import os
import pickle
import logging
from datetime import datetime
import numpy as np
import argparse
from datetime import datetime
from itertools import islice
import torch
from langchain.tools import Tool
import tiktoken
from multiprocessing import Process, Queue
import multiprocessing as mp
from difflib import unified_diff
import nltk
from collections import defaultdict
from openai import BadRequestError, OpenAIError, OpenAI
import openai
import json
import re
import logging
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import random
from prompt_selection import Demon_sampler
# ---- 日志配置----
log_filename = f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
# ensure punkt available once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class ChatGPT:
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.token_num = 0

    def get_response(self, input_text, turn_type):
        """
        这是主方法，用于处理输入文本并获取 LLM 的响应。
        根据是否为 debug 模式决定：
        ✅ 调试模式（debug = True）：
        创建用户消息后显示提示，等待手动输入模拟的返回值。
        🚀 正常运行模式：
        调用 query_API_to_get_message() 发起真实 API 请求。
        将用户输入与模型返回都追加到历史消息中。
        """

        if self.args.debug:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message.content)
            print("query API to get message:\n%s" % message.content)
            response = input("input the returned response:")
        else:
            message = self.create_message(input_text, turn_type)
            self.history_messages.append(message)
            self.history_contents.append(message['content'])
            message = self.query_API_to_get_message(self.history_messages)
            # return res.choices[0].message
            self.history_messages.append(message)
            self.history_contents.append(message.content)
            response = message.content.strip()
        return response

    def create_message(self, input_text, turn_type):
        """
        构造用户消息：create_message(input_text, turn_type)通过不同的 turn_type 构建多种提示类型的输入消息。
          turn_type 决定使用的模板段，如：
         "init_query": 使用初始化指令。
         "first_give_demonstration": 提供初始示例。
         "analogy_demonstration" / "supplement_demonstration": 类比或补充示例。
         "final_query_template": 用于最终提问，包含候选实体及问题。
         "directly_ask": 直接提问的模板。
        返回值为：{'role': 'user', 'content': input_text}
        """
        # res = self.LLM.get_response((question_text, answer), "query_generation")
        if turn_type == "first_give_demonstration":
            template = self.prompt['first_give_demonstration']
            question = input_text
            input_text = template.format(question=question)
        elif turn_type == "analogy_demonstration":
            template = self.prompt['analogy_demonstration']
            analogy_demons = input_text
            input_text = template.format(selected_analogy_demonstrations=analogy_demons)
        elif turn_type == "supplement_demonstration":
            template = self.prompt['supplement_demonstration']
            supplement_demons = input_text
            input_text = template.format(selected_supplement_demonstrations=supplement_demons)
        elif turn_type == "init_draft":
            template = self.prompt['init_draft']
            question, can_ents = input_text
            input_text = template.format(question=question, order_of_candidate=can_ents)
        elif turn_type == "query_generation":
            template = self.prompt['query_generation']
            question, answer = input_text
            input_text = template.format(question=question, answer=answer)
        elif turn_type == "revise":
            template = self.prompt['revise']
            question, answer, content = input_text
            input_text = template.format(content=content, question=question, answer=answer)
        elif turn_type == "final_query_template":
            template = self.prompt['final_query_template']
            origin_candidates_text,question_text,answer = input_text
            input_text = template.format(order_of_candidate=origin_candidates_text, question=question_text, CoT=answer)
        elif turn_type == "directly_ask":
            template = self.prompt['directly_ask']
            question = input_text
            input_text = template.format(question=question)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def count_tokens(self, messages, model="gpt-3.5-turbo"):
        """统计一组 messages 的 token 数"""
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")  # fallback

        tokens = 0
        for msg in messages:
            tokens += 4  # role + metadata
            for key, value in msg.items():
                tokens += len(enc.encode(value))
        tokens += 2  # 每个 reply 额外 tokens
        return tokens

    def query_API_to_get_message(self, messages):
        max_retries = 5
        retries = 0
        while True:
            try:
                client = OpenAI(api_key='sk-JMU1FWOBnZra7gVz1DRV8MlmNT7yCVr4VRB4PP539c5qRS80',
                                base_url='https://api.deepbricks.ai/v1/')
                #
                # tokens = self.count_tokens(messages, model="gpt-4o-mini")
                # print('message', messages)
                # print(f"[INFO] tokens={tokens}")

                res = client.chat.completions.create(
                    # model="o3-mini",
                    model="gpt-3.5-turbo",
                    # model="o4-mini",
                    # model="deepseek-R1-0528",
                    # model="GPT-5-chat",
                    messages=messages,
                    temperature=0,
                    max_tokens=self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )

                if self.args.debug_online:
                    print(res)

                self.token_num = res.usage.total_tokens
                return res.choices[0].message
            except Exception as e:
                print(f"OpenAI API error: {e}")

    def reset_history(self):
        self.history_messages = []
        self.history_contents = []
        self.token_num = 0

    def reset_history_messages(self):
        self.history_messages = []

    def reset_history_contents(self):
        self.history_contents = []

    def load_prompt_template(self, prompt_path, prompt_name):
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]


class RAT:
    def __init__(self, args):
        # RAT Pipeline
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=self.args.prompt_path, prompt_name=self.args.prompt_name,
                           max_tokens=self.args.max_tokens)
        self.newline_char = '\n'
        self.max_llm_input_token = self.args.max_llm_input_tokens
        self.prompt_selector = Demon_sampler(self.args)

        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []

        self.id2ent = defaultdict(str)
        self.ent2id = defaultdict(str)
        self.rel2id = defaultdict(str)
        self.ent2text = defaultdict(str)
        self.all_candidate_answers = defaultdict(list)
        self.align_text = defaultdict(str)

        self.supporting_file_path = self.args.supporting_file_path
        self.chunk_size = self.args.chunk_size
        self.chunk_overlap = self.args.chunk_overlap
        # GPU 加速
        self.embedding_model = SentenceTransformer(
            '/root/miniconda3/envs/shent/models/msmarco-distilbert-base-v4',
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # self.embedding_model = SentenceTransformer('/root/miniconda3/envs/shent/models/msmarco-distilbert-base-v4')
        self.encoding_name = "cl100k_base"

        self.chunks = self.load_and_chunk_text()
        # self.index, self.embeddings, self.chunks_search = self.build_faiss_index(self.chunks)
        # 构建或加载缓存好的 index
        self.index, self.chunks_search = self.build_faiss_index(
            self.chunks,
            save_path=f"cache/{self.args.dataset}_faiss"
        )
        # nltk.download('punkt')

        self.load_rel_txt_to_id()
        self.load_ent_map_id()
        self.load_all_candidate_answers()  ###得到所有的候选三元组
        self.load_ent_to_text()
        if self.args.align_text:
            self.load_align_text()

    def load_and_chunk_text(self):
        """按 chunk_size 加载文本"""
        with open(self.supporting_file_path, 'r') as f:
            text = f.read()

        step = self.chunk_size - self.chunk_overlap
        chunks = [text[i:i + self.chunk_size] for i in range(0, len(text), step)]
        return chunks

    def build_faiss_index(self, chunks, save_path="./outputs/faiss_index"):
        """支持缓存的 FAISS index 构建"""

        index_path = save_path + ".index"
        chunks_path = save_path + "_chunks.pkl"
        emb_path = save_path + "_emb.npy"

        # 如果缓存存在 → 直接加载
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            # print(f"[INFO] Loading cached FAISS index from {save_path}")
            logging.info(f" Loading cached FAISS index from {save_path}")
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)
            # embeddings = np.load(emb_path)
            return index, chunks


        logging.info(f" Building FAISS index for {len(chunks)} chunks...")
        embeddings = self.embedding_model.encode(
            chunks,
            batch_size=512,  # 调大
            convert_to_numpy=True,
            show_progress_bar=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        # 保存缓存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        faiss.write_index(index, index_path)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        np.save(emb_path, embeddings)

        # print(f"[INFO] FAISS index saved to {save_path}")
        logging.info(f" FAISS index saved to {save_path}")
        return index, embeddings, chunks

    # 检索相似的文本块
    def retrieve_similar_chunks(self, query):
        """检索前K个最相关的文本块"""
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding)
        distances, indices = self.index.search(np.array(query_embedding), self.args.k)
        return [self.chunks_search[i] for i in indices[0]if i >= 0]


    def get_content(self, query):  # query_list 是一个 JSON array
        """
        从本地知识库中进行 RAG 检索。
        query_list: list[str]，例如 ["RoboCop film language", "English Language", "American English"]
        返回：分 chunk 的文本列表
        """
        all_chunks = []
        # 如果 query 是字符串，就尝试解析成 JSON array
        if isinstance(query, str):
            try:
                query_list = json.loads(query)
            except json.JSONDecodeError:
                # 如果不是合法 JSON，就退化为单一字符串查询
                query_list = [query]
        elif isinstance(query, list):
            query_list = query
        else:
            raise ValueError(f"Unsupported query type: {type(query)}")

        logging.info(f"Searching queries: {query_list}")

        for q in query_list:
            logging.info(f"Searching for query: {q}")
            top_chunks = self.retrieve_similar_chunks(q)
            if top_chunks:
                all_chunks.extend(top_chunks)
            else:
                logging.info(f">>> No good Knowledge Graph Search Result was found for query: {q}")

        if not all_chunks:
            return None

        # 去重
        all_chunks = list(set(all_chunks))

        # 拼接为整体文本
        retrieved_text = " ".join(all_chunks)

        # 切分成更小的段落
        trunked_texts = self.chunk_text_by_sentence(retrieved_text, 1500)
        trunked_texts = [trunked_text.replace('\n', " ") for trunked_text in trunked_texts]

        return trunked_texts

    def relation_text(self, relation, align_text):
        if align_text=="True":
            return self.align_text[relation]
        else:
            relation_hierachy_list = relation.strip().replace('.',' ').split('/')
            final_string = ''
            for st in reversed(relation_hierachy_list):
                if st != "":
                    final_string += st + " of "
            return final_string

    def count_token(self, string):
        """Returns the number of tokens in a text string using tiktoken."""
        if string is None or string == "":
            return 0
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


    def forward(self, relation, entity):
        self.LLM.reset_history()
        self.reset_history()

        ##得到candidate的id
        ent_str = self.ent2text[entity]
        candidate_ids = self.all_candidate_answers['\t'.join([str(self.ent2id[entity]), str(self.rel2id[relation])])]

        ##得到candidate的text
        for id in candidate_ids[:self.args.candidate_num]:
            self.candidate_answers.append(self.ent2text[self.id2ent[str(id)]])
        origin_candidates_text = self.serialize_candidate_answers()

        ##得到demonstration_text
        question_text = ''
        if self.args.query == 'tail':
            question_text = self.generate_demonstration_text((ent_str, relation, ''))
        elif self.args.query == 'head':
            question_text = self.generate_demonstration_text(('', relation, ent_str))

        current_demon_response = self.LLM.get_response((question_text),"first_give_demonstration")

        ###true_demons: [[tpe_text,question,t1_text],[tpe_text,question,t2_text],...]
        true_demons = self.prompt_selector.true_candidate_v2(entity, relation, num=args.demon_per_step // 2)

        ####得到序列化之后的prompt中的所有真实三元组案例组成的一段text：true_demon_text，注意这里面都是真实答案的，也即在序列化时t!= ''
        true_demon_text = self.serialize_demonstrations(true_demons)
        if true_demon_text != "None.":
            # 该步骤放入prompt的是当前query的relation-entity在训练集和验证集中的答案集。该步骤的目的应该是在最开始优先在prompt中放与当前query最相关的三元组信息。
            current_demon_response = self.LLM.get_response((true_demon_text), "analogy_demonstration")
        if self.LLM.token_num >= args.max_llm_input_tokens - 1000:
            self.LLM.history_messages.pop()
            self.LLM.history_messages.pop()
            self.LLM.history_contents.pop()
            self.LLM.history_contents.pop()


        ### RAT pipline
        logging.info(f"{datetime.now()} Obtaining Draft ...")
        ####llm排序
        draft = self.LLM.get_response((question_text, origin_candidates_text), "init_draft")
        logging.info(f"{datetime.now()}  Returning Draft")
        logging.info(f"Draft: {draft}")

        logging.info(f"{datetime.now()}  Handling Draft ...")
        draft_paragraphs = self.split_draft(draft)
        logging.info(f"{datetime.now()}  The draft has been split into {len(draft_paragraphs)} parts")
        answer = ""

        for i, p in enumerate(draft_paragraphs):
            logging.info(f"{datetime.now()}  Revise the {i + 1}/{len(draft_paragraphs)}th part...")

            ###不断携带前面的历史答案进行修正
            answer = answer + '\n\n' + p
            ####查看p有几句话，或者改成尤其是最后一段而不是尤其是最后一段话
            logging.info(f"{datetime.now()}  Generating Corresponding Query...")
            ###得到与该部分内容相关的query
            res = self.LLM.get_response((question_text, answer), "query_generation")
            if hasattr(res, "content"):
                    query = res.content
            elif isinstance(res, str):
                    query = res
            else:
                logging.warning(f"Unexpected response type: {type(res)}. Skipping...")
                continue

            if self.LLM.token_num >= self.args.max_llm_input_tokens:
                self.LLM.history_messages.pop()
                self.LLM.history_messages.pop()
                self.LLM.history_contents.pop()
                self.LLM.history_contents.pop()
                break
            logging.info(f">>> {i}/{len(draft_paragraphs)} Query: {query.replace(self.newline_char, ' ')}")  ##str.replace(old, new[, count])

            logging.info(f"{datetime.now()}  Obtaining Knowledge graph content...")
            ###根据query来search到相关的page
            result = self.get_content(query)
            if not result:
                logging.info(f"{datetime.now()}  Skip the subsequent steps...")
                continue
            else:
                content = result
            for j, c in enumerate(content):
                if j > 3:
                    break
                logging.info(f"{datetime.now()}  Revise corresponding content according to local knowledge graph...[{j}/{min(len(content), 3)}]")

                res = self.LLM.get_response((question_text, answer, c), "revise")
                if not res:
                    logging.info(f"{datetime.now()}  Skip the subsequent steps...")
                    continue
                else:
                    answer = res
                logging.info(f"{datetime.now()}  Revise Finished [{j}/{min(len(content), 3)}]")
                if self.LLM.token_num >= self.args.max_llm_input_tokens-1000:
                # if self.LLM.token_num >= self.args.max_llm_input_tokens - query_token_num:
                    self.LLM.history_messages.pop()
                    self.LLM.history_messages.pop()
                    self.LLM.history_contents.pop()
                    self.LLM.history_contents.pop()
                    break
        final_response = self.LLM.get_response((origin_candidates_text,question_text,answer), "final_query_template")
        self.log.append(final_response)
        final_order = self.parse_result(final_response, entity, relation)
        self.log.append(final_order)
        used_tokens = self.LLM.token_num
        return final_order, self.LLM.history_contents, self.log, used_tokens

    def chunk_text_by_sentence(self, text, chunk_size=2048):
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)  # 英文 + 中文符号
        chunked_text = []
        curr_chunk = []

        for sentence in sentences:
            if self.count_token(" ".join(curr_chunk)) + self.count_token(
                    sentence) + 2 <= chunk_size:
                curr_chunk.append(sentence)
            else:
                if curr_chunk:
                    chunked_text.append(". ".join(curr_chunk))
                curr_chunk = [sentence]

        if curr_chunk:
            chunked_text.append(". ".join(curr_chunk))
        return chunked_text


    def serialize_candidate_answers(self):
        """
        将多个三元组或实体文本拼接成字符串，用于插入到最终 prompt 模板中
        """
        candidiate_str = '[' + ','.join(self.candidate_answers) + ']'
        return candidiate_str

    def serialize_demonstrations(self, demon_triples):  ###得到将所有三元组进行序列化之后的text
        """
        将多个三元组或实体文本拼接成字符串，用于插入到最终 prompt 模板中
        """
        demon_text = ""
        for tp in demon_triples:
            demon_text += self.generate_demonstration_text(tp) + '. '
        demon_text.strip()
        if demon_text == "": demon_text = "None."
        return demon_text

    def generate_demonstration_text(self, triple):
        """

       将三元组格式化为自然语言 prompt（以 mask 表达）
       用于构建 Prompt 中的 few-shot 示例，如：
       predict the tail entity [MASK] from the given (Barack Obama, place of birth, [MASK]) by completing the sentence
       "what is the place of birth of Barack Obama? The answer is ... "
       """
        h, r, t = triple
        demonstration_text = ""
        if self.args.query == 'tail':
            if self.args.align_text:
                demonstration_text = 'predict the tail entity [MASK] from the given ('
                demonstration_text += h + ', ' + self.relation_text(r, False)
                demonstration_text += ", [MASK]) by completing the sentence \""
                demonstration_text += (self.relation_text(r, True).replace("[H]", h).
                                       replace("[T]", "[the answer]") + '? The answer is \"')
                if t != '':
                    demonstration_text += ". The answer is " + t + ", so the [MASK] is " + t
            else:
                demonstration_text = 'predict the tail entity [MASK] from the given ('
                demonstration_text += h + ', ' + self.relation_text(r, False)
                demonstration_text += ", [MASK]) by completing the sentence \"what is the "
                demonstration_text += self.relation_text(r, False) + h + '? The answer is \"'
                if t != '':
                    demonstration_text += ". The answer is " + t + ", so the [MASK] is " + t
        elif self.args.query == 'head':
            if self.args.align_text:
                demonstration_text = 'predict the head entity [MASK] from the given ('
                demonstration_text += '[MASK]' + ', ' + self.relation_text(r, False)
                demonstration_text += ", " + t + ") by completing the sentence \""
                demonstration_text += self.relation_text(r, True).replace("[H]", "[the answer]").replace("[T]",
                                                                                                         t) + '? The answer is \"'
                if h != '':
                    demonstration_text += ". The answer is " + h + ", so the [MASK] is " + h
            else:
                demonstration_text = 'predict the head entity [MASK] from the given ('
                demonstration_text += '[MASK]' + ', ' + self.relation_text(r, False)
                demonstration_text += ", " + t + ") by completing the sentence \"" + t + " is the "
                demonstration_text += self.relation_text(r, False) + "what" + '? The answer is \"'
                if h != '':
                    demonstration_text += ". The answer is " + h + ", so the [MASK] is " + h
        return demonstration_text

    def parse_result(self, response, entity, relation):
        response = response.lower()
        candidate_answers = []
        ##得到candidate的id
        candidate_ids = self.all_candidate_answers['\t'.join([str(self.ent2id[entity]), str(self.rel2id[relation])])]

        ##得到candidate的text
        for id in candidate_ids[:self.args.candidate_num]:
            candidate_answers.append(self.ent2text[self.id2ent[str(id)]])

        if "the final order:" in response:
            final_order_raw = re.split("the final order:", response)[1].strip().strip('.').strip('\[').strip('\]')
            final_order_raw_list = final_order_raw.split(' | ')
            final_order_list = []
            for candidate in final_order_raw_list:
                if candidate not in final_order_list:
                    final_order_list.append(candidate)
            final_order = ' | '.join(final_order_list)
        else:
            final_order = ' | '.join(candidate_answers)

        return final_order

    def check_work_flow(self, response):
        if "no" in response.lower():
            return False
        return True

    def reset_history(self):
        self.log = []
        self.candidate_answers = []
        self.selected_demonstrations = []

    def load_all_candidate_answers(self):
        with open("/root/autodl-tmp/dataset/" + self.args.dataset + "/retriever_candidate_" + self.args.query + ".txt", 'r') as load_f:
            self.all_candidate_answers = json.load(load_f)  # 每个问题有100个candidates

    def load_align_text(self):
        with open("/root/autodl-tmp/dataset/" + self.args.dataset + "/alignment/alignment_clean.txt", 'r') as load_f:
            self.align_text = json.load(load_f)

    def load_rel_txt_to_id(self):
        with open('/root/autodl-tmp/dataset/' + self.args.dataset + '/get_neighbor/relation2id.txt', 'r') as file:
            relation_lines = file.readlines()
            for line in relation_lines:
                _name, _id = line.strip().split("\t")
                self.rel2id[_name] = _id

    def load_ent_map_id(self):
        with open('/root/autodl-tmp/dataset/' + self.args.dataset + '/get_neighbor/entity2id.txt', 'r') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                _name, _id = line.strip().split("\t")
                self.ent2id[_name] = _id
                self.id2ent[_id] = _name

    def load_ent_to_text(self):
        with open('/root/autodl-tmp/dataset/' + self.args.dataset + '/entity2text.txt', 'r') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text

    def split_draft(self, draft, split_char='###'):
        # 将draft切分为多个段落
        draft_paragraphs = [p.strip() for p in draft.split(split_char) if p.strip()]
        return draft_paragraphs


def main(args, all_data, idx, api_key):
    """
    在测试集test.answer.txt上运行rat来得到每一个测试实例的预测结果，并将结果保存在output文件中
    """
    openai.api_key = api_key
    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:  # 多线程
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    logging.info(f"Start PID {os.getpid()} and save to {output_path}")
    rat = RAT(args)

    count = 0
    valid_count = 0
    all_used_tokens = []

    with open(output_path, "w") as f:
        with open(chat_log_path, "w") as fclog:
            for sample in tqdm(all_data, total=len(all_data)):  ##all_data：测试集test.tsv
                count += 1
                # sample:
                # {"ID": 2477,
                #   "HeadEntity": "/m/0299hs",
                #   "Answer": "/m/02h40lc",
                #   "Question": "/film/film/language"}
                try:
                    tpe = sample['HeadEntity'] if args.query == 'tail' else sample['Answer']
                    question = sample['Question']  ###relation

                    prediction, chat_history, record,used_tokens = rat.forward(question, tpe)  ###输入关系和头实体（尾实体预测）或尾实体（预测头实体）
                    valid_count += 1
                    all_used_tokens.append(used_tokens)

                except BadRequestError as e:  # 对应旧版的 InvalidRequestError
                    print(e)
                    continue
                except OpenAIError as e:  # 捕获其他 OpenAI 相关错误
                    logging.exception(e)
                    continue
                except Exception as e:  # 捕获非 OpenAI 相关的错误
                    logging.exception(e)
                    continue


                chat = str(sample["ID"]) + "\n" + "\n******\n".join(chat_history) + "\nAnswers: " + str(
                    sample['Answer']) + "\n------------------------------------------\n"
                fclog.write(chat)

                sample["Prediction"] = prediction
                f.write(json.dumps(sample) + "\n")

    avg_tokens = sum(all_used_tokens) / len(all_used_tokens)
    print(f"Average tokens per instance: {avg_tokens:.2f}")

    logging.info("---------------PID %d end with %d/%d samples--------------" % (os.getpid(), valid_count, count))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="fb15k-237")
    parser.add_argument('--candidate_num', default=50, type=int)
    parser.add_argument('--output_path', default="./outputs/fb15k-237/output_tail.txt")#fb15k-237
    parser.add_argument('--chat_log_path', default="./outputs/fb15k-237/chat_tail.txt")
    parser.add_argument('--query', default="tail")
    parser.add_argument('--model_path', default=None)

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--debug_online', action="store_true")
    parser.add_argument('--align_text', default="False")

    parser.add_argument('--max_tokens', default=1500, type=int, help='max-token')
    parser.add_argument('--prompt_path', default="./prompts/icl_rat.json")
    parser.add_argument('--prompt_name', default="chat", )
    parser.add_argument('--bagging_type', default="llm", )
    parser.add_argument('--overwrite', action="store_true")
    parser.add_argument('--device', default=0, help='the gpu device')

    parser.add_argument('--api_key', default="sk-JMU1FWOBnZra7gVz1DRV8MlmNT7yCVr4VRB4PP539c5qRS80", type=str)
    parser.add_argument('--demon_per_step', default=10)
    parser.add_argument('--eff_demon_step', default=10)
    parser.add_argument('--max_demon_step', default=10)
    parser.add_argument('--max_llm_input_tokens', default=5000, type=int)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')

    parser.add_argument('--supporting_file_path', default="/root/autodl-tmp/dataset/fb15k-237/description/supporting_file.txt")
    parser.add_argument('--chunk_size', default=500)
    parser.add_argument('--chunk_overlap', default=100)
    parser.add_argument('--k', default=1)
    parser.add_argument("--entity_file", default="/root/autodl-tmp/dataset/fb15k-237/entity2text.txt")
    parser.add_argument("--filter_file", default="/root/autodl-tmp/dataset/fb15k-237/filter_head.txt")

    args = parser.parse_args()
    args.output_path = './outputs/' + args.dataset + '/output_' + args.query + '_rat-context.txt'
    args.chat_log_path = './outputs/' + args.dataset + '/chat_' + args.query + '_rat-context.txt'
    logging.info("Start querying the LLM.")
    return args
def merge_outputs(args):
    """
    将多个子进程的结果文件合并成最终文件
    """
    final_output = args.output_path
    final_chatlog = args.chat_log_path

    with open(final_output, "w") as fout, open(final_chatlog, "w") as fchat:
        for fname in sorted(os.listdir(os.path.dirname(final_output))):
            if fname.startswith(os.path.basename(final_output)):
                with open(os.path.join(os.path.dirname(final_output), fname)) as f:
                    for line in f:
                        fout.write(line)
            if fname.startswith(os.path.basename(final_chatlog)):
                with open(os.path.join(os.path.dirname(final_chatlog), fname)) as f:
                    for line in f:
                        fchat.write(line)

    logging.info(f"✅ 合并完成：{final_output}, {final_chatlog}")


if __name__ == '__main__':
    args = parse_args()
    random.seed(42)

    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)

    with open("/root/autodl-tmp/dataset/" + args.dataset + "/select_sample.txt", 'r') as load_f:  ###加载test的三元组
        test_triplet = json.load(load_f)

    test_triplet = test_triplet[:6]

    logging.info("Totally %d test examples." % len(test_triplet))


    if args.debug_online:
        test_triplet = test_triplet[0:2 * args.num_process]

    if args.num_process == 1:
        main(args, test_triplet, idx=-1, api_key=args.api_key)
    else:
        num_each_split = int(len(test_triplet) / args.num_process)
        p = mp.Pool(args.num_process)
        for idx in range(args.num_process):
            start = idx * num_each_split
            if idx == args.num_process - 1:
                end = max((idx + 1) * num_each_split, len(test_triplet))
            else:
                end = (idx + 1) * num_each_split
            split_data = test_triplet[start:end]

            if args.api_key.startswith("sk-"):
                api_key = args.api_key
            else:
                api_key = all_keys[idx]

            try:
                p.apply_async(main, args=(args, split_data, idx, api_key))
            except Exception as e:
                logging.exception(e)

        p.close()
        p.join()
        merge_outputs(args)
        logging.info("All of the child processes over!")




    # predictions = load_predictions(args.output_path)
    # entity_text = load_ent_to_text(args.entity_file)
    # filter_tail = load_filter_tail(args.filter_file)
    #
    # metrics = evaluate_ranking(predictions, entity_text, filter_tail)
    # print("=== Evaluation Results ===")
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")


