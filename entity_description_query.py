import argparse
import json
import multiprocessing as mp
import numpy as np
from openai import BadRequestError, OpenAIError
import logging
import os
import openai
from tqdm import tqdm
import simplejson
import heapq
from openai import OpenAI
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


class ChatGPT:
    ####构造函数 __init__()
    # 初始化模型、读取 prompt 模板、配置最大 token 长度等。
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.token_num = 0
    
    def get_response(self, input_text, turn_type):
        # print("input_text",input_text)
        # 核心函数 get_response()
        # 根据输入文本构建完整消息，并调用 OpenAI 接口获得回复，或在 debug 模式下由用户输入模拟回复。
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
            self.history_messages.append(message)
            # self.history_contents.append(message['content'])
            self.history_contents.append(message.content)
            response = message.content.strip()
        return response

    def query_localLLM_to_get_response(self,message):
        # input: message: {role': 'user', 'content': string(input_text to LLM, which has implemented) }
        # return:  response: {role': 'assistant', 'content': string(output_text which need you to fetch and store here)}
        output_text = "" #modifiy here
        response = {'role': 'assistant', 'content': output_text}
        if output_text == "":
            print("Implement The function")
        return response
    
    def create_message(self, input_text, turn_type):
        # 构建一轮对话中，用户输入的完整 prompt（插入演示示例和目标关系文本）。
        if turn_type == "init_query":  
            template = self.prompt['init_query']
            ent, triples_str = input_text
            input_text = template.format(entity=ent, triple_str=triples_str)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages):
        # 使用 OpenAI 的 ChatCompletion.create() 方法请求 GPT 模型，支持自动重试策略。
        while True:
            try:

                client = OpenAI(api_key='sk-JMU1FWOBnZra7gVz1DRV8MlmNT7yCVr4VRB4PP539c5qRS80',
                                base_url='https://api.deepbricks.ai/v1/')
                print('message', messages)

                res = client.chat.completions.create(
                    model="o3-mini",
                    # model="gpt-4-turbo",
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
        ###加载需要的prompt
        if prompt_path.endswith(".json"):
            with open(prompt_path, "rb") as f:
                prompt = json.load(f)
            return prompt[prompt_name]
        
        
from collections import defaultdict
class Solver:
    def __init__(self, args):
        self.args = args
        self.LLM = ChatGPT(args=args, prompt_path=args.prompt_path, prompt_name=args.prompt_name,
                           max_tokens=args.max_tokens)

        self.log = []

        self.ent2text = defaultdict(str)
        self.rel2text = defaultdict(str)
        self.rel2align_text = defaultdict(str)
        self.ent2triples = defaultdict(list)
        self.rel2triples = defaultdict(list)
        self.entity2description_clean = defaultdict(str)
        self.ent2paths = defaultdict(str)

        self.load_ent_to_text()
        self.load_rel_to_text()
        self.supporting_text_rel()
        self.supporting_triple("dataset/" + args.dataset + "/train.tsv")
        self.supporting_triple("dataset/" + args.dataset + "/dev.tsv")
        self.generate_path()
        # self.data_combine()

                
    def forward(self, ent): #Here tpe_id not a int id, but like '/m/08966'
        self.LLM.reset_history()
        self.reset_history()
        triples_str = self.triple_selector(ent)

        final_response = self.LLM.get_response((self.ent2text[str(ent)], triples_str),"init_query") ###得到关系的文本化描述
        self.log.append(final_response)

        return final_response, self.LLM.history_contents



    def parse_description(self, str):  ###解析出关系的干净的纯文本描述
        description = str
        lower_text = description.lower()
        if "**search text**:" in lower_text:
            st_idx = lower_text.index("**search text**:") + len("**search text**:")
            search_text = description[st_idx:].strip()
        elif "search text:" in lower_text:
            st_idx = lower_text.index("search text:") + len("search text:")
            search_text = description[st_idx:].strip()
        else:
            # 如果格式不符，全部当raw_text返回
            search_text = None
        return search_text

    def parse_result(self):
        with open("dataset/" + self.args.dataset + "/description/description_output.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                description_text = simplejson.loads(line)
                self.entity2description_clean[description_text["Raw"]] = self.parse_description(description_text["Description"])

        with open("dataset/" + args.dataset + "/description/description_clean.txt", "w") as f:
            f.write(json.dumps(self.entity2description_clean, indent=1))
    def reset_history(self):
        self.log = []
    
    def load_ent_to_text(self):
        with open('dataset/' + self.args.dataset + '/entity2text.txt', 'r') as file:
            entity_lines = file.readlines()
            for line in entity_lines:
                ent, text = line.strip().split("\t")
                self.ent2text[ent] = text
    def load_rel_to_text(self):
        with open('dataset/' + self.args.dataset + '/relation2text.txt', 'r') as file:
            rel_lines = file.readlines()
            for line in rel_lines:
                rel, text = line.strip().split("\t")

                if "/" in text:
                    text = text.replace("/", " > ").replace(".", "").strip()
                self.rel2text[rel] = text
        self.rel2text = {k.strip('"'): v for k, v in self.rel2text.items()}

    def supporting_text_rel(self):
        with open(self.args.alignment_path, 'r') as f:
            alignment_dict = json.load(f)

        for rel in self.rel2text:
            desc = alignment_dict.get(rel, "")
            self.rel2align_text[rel] = desc

    def supporting_triple(self,file):
        """
        输入：三元组文件路径（默认为 train.tsv）
        输出：
            supporting_triple_dict_entity - dict[str, list[tuple[str, str, str]]]
            supporting_triple_dict_relation - dict[str, list[tuple[str, str, str]]]
        """
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue  # 跳过格式错误的行

                head, relation, tail = parts

                triple = (head, relation, tail)
                # 添加到实体字典（head 和 tail 都是实体）
                for entity in [head, tail]:
                    if entity not in self.ent2triples:
                        self.ent2triples[entity] = []
                    self.ent2triples[entity].append(triple)

                # 添加到关系字典
                if relation not in self.rel2triples:
                    self.rel2triples[relation] = []
                self.rel2triples[relation].append(triple)

    def generate_path(self):
        # Step 1: 构建有向图
        graph = defaultdict(list)  # ent1: list of (relation, ent2)

        def load_triples(file):
            with open(file, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3:
                        ent1, rel, ent2 = parts
                        graph[ent1].append((rel, ent2))

        load_triples("dataset/" + self.args.dataset +"/train.tsv")
        load_triples("dataset/" + self.args.dataset +"/dev.tsv")

        # Step 2: DFS 查找所有路径（最多4步关系 = 9个元素）
        entity_to_paths = defaultdict(list)

        def dfs(current_node, path, visited_entities):
            current_length = len(path) // 2  # 路径长度按边数计算
            if current_length == 3 or current_length == 5:  # 边数为3(路径长度5)或5(路径长度7)
                entity_to_paths[path[0]].append(path)  # 将路径记录到起始实体

            if current_length >= 5:  # 超过最大需要的长度就停止
                return

            for rel, neighbor in graph.get(current_node, []):
                if neighbor in visited_entities:
                    continue  # 去环

                new_path = path + [rel, neighbor]
                new_visited = visited_entities | {neighbor}
                dfs(neighbor, new_path, new_visited)

        # Step 3: 对图中所有实体执行 DFS
        for start_entity in graph:
            dfs(start_entity, [start_entity], {start_entity})

        for entity, paths in entity_to_paths.items():

            texts = [" -> ".join(p) for p in paths]
            path_sentences = '. '.join(texts)
            self.ent2paths[entity] = path_sentences
    def triple_selector(self,entity):

        selected_triples = self.poolsampler_v2(entity)
        true_triple_str = self.serialize_triples(selected_triples)
        return true_triple_str

    def poolsampler_v2(self, entity):

        sorted_list = self.Diversity_arranged_v2(entity)
        # print(self.args.select_triples,sorted_list)
        selected_list = sorted_list[:self.args.select_triples]
        # print('selected_list',selected_list)
        return selected_list

    def Diversity_arranged_v2(self, tpe):
        ##该函数的目的是：对给定实体tpe对应的三元组列表demon_list进行重排序，
        # 使得最终排序后的三元组列表中实体尽可能多样化（即尽量避免相同的实体重复出现）
        ''' 应用场景（举例）：在构造知识图谱的类比推理训练样本时：
          不希望某个实体如 "Germany" 或 "Barack Obama" 在多条样本中频繁出现；
          这个函数可以自动排序，使不同实体分布更均衡，从而提升训练或评估的泛化能力。
        '''
        demon_list = self.ent2triples[tpe]
        entity_counter = defaultdict(int)

        def count_sum(triple):
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                raise ValueError(f"Invalid triple format: {triple}")
            return entity_counter[triple[0]] + entity_counter[triple[2]], triple

        # 初始队列
        priority_queue = [count_sum(triple) for triple in demon_list]
        heapq.heapify(priority_queue)

        sorted_list = []
        while priority_queue:
            _, next_triple = heapq.heappop(priority_queue)
            sorted_list.append(next_triple)
            entity_counter[next_triple[0]] += 1
            entity_counter[next_triple[2]] += 1

            # 重新计算剩余元素的优先级，只取 triple 部分
            priority_queue = [count_sum(triple) for _, triple in priority_queue]
            heapq.heapify(priority_queue)

        return sorted_list

    def serialize_triples(self, demon_triples):  ###得到将所有三元组进行序列化之后的text
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
        h,r,t = triple
        h = self.ent2text[h]
        t = self.ent2text[t]
        if self.args.mode == 'no_alignment': triples_text = "\""+ t +"\"" + " " + self.rel2text[r].strip('"') + " " + "\""+ h +"\""
        else: triples_text = self.rel2align_text[r].replace("[H]", "\""+ h +"\"").replace("[T]", "\""+ t +"\"")
        return triples_text
    def data_combine(self):
        """
        整合实体和关系的信息并保存为 JSON 文件。
        """

        supporting_info_e = ''
        supporting_info_r = ''

        # 构建实体信息字典
        for entity, text in self.ent2text.items():
            supporting_info_e += entity + ":" +text +"\n"
            supporting_info_e += "corresponding description:" + self.entity2description_clean.get(entity, "") + "\n"

            supporting_info_e += "related triples:" + ", ".join(map(str, self.ent2triples.get(entity, []))) + "\n"
            supporting_info_e += "related paths:" + self.ent2paths.get(entity, "") + ".\n"


        # 构建关系信息字典
        for relation, text in self.rel2text.items():
            supporting_info_r += relation + ":" + text + "\n"
            supporting_info_r += "corresponding description:" + self.rel2align_text.get(relation, "") + "\n"
            supporting_info_r += "related triples:" + ", ".join(map(str, self.rel2triples.get(relation, [])) )+ "\n"

        # 保存到文件
        with open(self.args.supporting_file_path, 'w') as f:
            f.write(supporting_info_e)
            f.write(supporting_info_r)
        print(f"The supporting information of entity & relation has been saved to {self.args.supporting_file_path}")

    def data_combine_dict(self):
        """
        整合实体和关系的信息并保存为 JSON 文件。
        """

        supporting_info_e = {}
        supporting_info_r = {}

        # 构建实体信息字典
        for entity, text in self.ent2text.items():
            supporting_info_e[entity] = {
                "entity": entity,
                "name": text,
                "triples": self.ent2triples.get(entity, []),
                "description": self.entity2description_clean.get(entity, ""),
                "paths": self.ent2paths.get(entity, "")

            }

        # 构建关系信息字典
        for relation, text in self.rel2text.items():
            supporting_info_r[relation] = {
                "relation": relation,
                "name": text,
                "description": self.rel2align_text.get(relation, ""),
                "triples": self.rel2triples.get(relation, [])
            }

        # 保存到文件
        with open(self.args.supporting_file_path, 'w') as f:
            json.dump(supporting_info_e, f, ensure_ascii=False, indent=2)
            json.dump(supporting_info_r, f, ensure_ascii=False, indent=2)

        print(f"The supporting information of entity & relation has been saved to {self.args.supporting_file_path}")

def main(args, entities, idx, api_key):####得到关系r的text描述
    openai.api_key = api_key
    if idx == -1:
        output_path = args.output_path
        chat_log_path = args.chat_log_path
    else:
        idx = "0" + str(idx) if idx < 10 else str(idx)  # 00 01 02 ... 29
        output_path = args.output_path + "_" + idx
        chat_log_path = args.chat_log_path + "_" + idx

    print("Start PID %d and save to %s" % (os.getpid(), chat_log_path))
    solver = Solver(args)
    # tags=0
    # with open(output_path, "w") as f: ###得到/dataset/wn18rr/description/description_output.txt 文件
    #     with open(chat_log_path, "w") as fclog: ###得到/dataset/wn18rr/description/description_chat.txt 文件
    #         for ent in tqdm(entities.keys()):
    #             try:
    #                 clean_entity, chat_history = solver.forward(ent)
    #
    #             except BadRequestError as e:  # 对应旧版的 InvalidRequestError
    #                 print(e)
    #                 continue
    #             except OpenAIError as e:  # 捕获其他 OpenAI 相关错误
    #                 logging.exception(e)
    #                 continue
    #             except Exception as e:  # 捕获非 OpenAI 相关的错误
    #                 logging.exception(e)
    #                 continue
    #
    #             # print('clean_entity',clean_entity)
    #
    #             clean_text = defaultdict(str)
    #             clean_text["Raw"] = ent
    #             clean_text["Description"] = clean_entity ####最终得到的实体的描述文本，并存入文件
    #             f.write(json.dumps(clean_text) + "\n")
    #
    #             chat = str(ent) + "\n" + "\n******\n".join(chat_history) + "\n------------------------------------------\n"
    #             fclog.write(chat)

    # solver.parse_result()
    solver.data_combine_dict()

    print("---------------PID %d end--------------" % (os.getpid()))



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="wn18rr") #wn18rr,fb15k-237
    parser.add_argument('--output_path', default="./dataset/wn18rr/description/description_output.txt")
    parser.add_argument('--chat_log_path', default="./dataset/wn18rr/description/description_chat.txt")

    parser.add_argument('--mode', default="alignment")  # no_alignment or alignment
    parser.add_argument('--alignment_path', default="./dataset/wn18rr/alignment/alignment_clean.txt")
    parser.add_argument('--supporting_file_path', default="./dataset/wn18rr/supporting_file_dict.txt")

    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--debug_online', action="store_true")

    parser.add_argument('--select_triples', default=3, type=int,help='the number of selected triples')
    parser.add_argument('--prompt_path', default="./prompts/entity_description_fb.json")
    parser.add_argument('--prompt_name', default="chat", )
    parser.add_argument('--bagging_type', default="llm", )

    parser.add_argument('--device', default=0, help='the gpu device')

    parser.add_argument('--api_key', default="sk-JMU1FWOBnZra7gVz1DRV8MlmNT7yCVr4VRB4PP539c5qRS80", type=str)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')


    parser.add_argument('--max_tokens', default=500, type=int, help='max-token')
    args = parser.parse_args()

    print("Start querying the LLM.")
    return args
def run_worker(idx, args, sub_entities, api_key):
    """
    子进程调用 main()，处理一部分实体
    """
    main(args, sub_entities, idx, api_key)
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

    print(f"✅ 合并完成：{final_output}, {final_chatlog}")


if __name__ == '__main__':
    args = parse_args()
    print('ok!')

    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
        # 读取实体
    ent2text = defaultdict(str)
    with open('dataset/' + args.dataset + '/entity2text.txt', 'r') as file:
        entity_lines = file.readlines()
        for line in entity_lines:#[:10000]~0-999;[10000:20000]~10000-19999（含）;[20000:30000]~20000-29999;[30000:]~30000-40943
            ent, text = line.strip().split("\t")
            ent2text[ent] = text


    # 多进程模式
    if args.num_process > 1:
        entities = list(ent2text.keys())
        chunks = np.array_split(entities, args.num_process)

        procs = []
        for i, chunk in enumerate(chunks):
            sub_dict = {k: ent2text[k] for k in chunk}
            p = mp.Process(target=run_worker, args=(i, args, sub_dict, args.api_key))
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # 合并结果文件
        merge_outputs(args)

    else:
        # 单进程逻辑保持不变
        main(args, ent2text, idx=-1, api_key=args.api_key)








