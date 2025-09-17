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

class ChatGPT:
    
    def __init__(self, args, prompt_path, prompt_name, max_tokens):
        self.args = args
        self.history_messages = []
        self.history_contents = []
        self.max_tokens = max_tokens
        self.prompt = self.load_prompt_template(prompt_path, prompt_name)
        self.token_num = 0
    
    def get_response(self, input_text, turn_type):
       
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
        output_text = "" 
        response = {'role': 'assistant', 'content': output_text}
        if output_text == "":
            print("Implement The function")
        return response
    
    def create_message(self, input_text, turn_type):
        if turn_type == "init_query":  
            template = self.prompt['init_query']
            ent, triples_str = input_text
            input_text = template.format(entity=ent, triple_str=triples_str)
        else:
            raise NotImplementedError
        message = {'role': 'user', 'content': input_text}
        return message

    def query_API_to_get_message(self, messages):
        while True:
            try:
                client = OpenAI(api_key='Your API',
                                base_url='Your BASE_URL')
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

        final_response = self.LLM.get_response((self.ent2text[str(ent)], triples_str),"init_query")
        self.log.append(final_response)

        return final_response, self.LLM.history_contents



    def parse_description(self, str): 
        description = str
        lower_text = description.lower()
        if "**search text**:" in lower_text:
            st_idx = lower_text.index("**search text**:") + len("**search text**:")
            search_text = description[st_idx:].strip()
        elif "search text:" in lower_text:
            st_idx = lower_text.index("search text:") + len("search text:")
            search_text = description[st_idx:].strip()
        else:
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
       
        with open(file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) != 3:
                    continue  

                head, relation, tail = parts

                triple = (head, relation, tail)
                
                for entity in [head, tail]:
                    if entity not in self.ent2triples:
                        self.ent2triples[entity] = []
                    self.ent2triples[entity].append(triple)

                
                if relation not in self.rel2triples:
                    self.rel2triples[relation] = []
                self.rel2triples[relation].append(triple)

    def generate_path(self):
        
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

        
        entity_to_paths = defaultdict(list)

        def dfs(current_node, path, visited_entities):
            current_length = len(path) // 2 
            if current_length == 3 or current_length == 5: 
                entity_to_paths[path[0]].append(path)  

            if current_length >= 5: 
                return

            for rel, neighbor in graph.get(current_node, []):
                if neighbor in visited_entities:
                    continue 

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
        demon_list = self.ent2triples[tpe]
        entity_counter = defaultdict(int)

        def count_sum(triple):
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                raise ValueError(f"Invalid triple format: {triple}")
            return entity_counter[triple[0]] + entity_counter[triple[2]], triple

        priority_queue = [count_sum(triple) for triple in demon_list]
        heapq.heapify(priority_queue)

        sorted_list = []
        while priority_queue:
            _, next_triple = heapq.heappop(priority_queue)
            sorted_list.append(next_triple)
            entity_counter[next_triple[0]] += 1
            entity_counter[next_triple[2]] += 1

            priority_queue = [count_sum(triple) for _, triple in priority_queue]
            heapq.heapify(priority_queue)

        return sorted_list

    def serialize_triples(self, demon_triples):  
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
        supporting_info_e = ''
        supporting_info_r = ''

        for entity, text in self.ent2text.items():
            supporting_info_e += entity + ":" +text +"\n"
            supporting_info_e += "corresponding description:" + self.entity2description_clean.get(entity, "") + "\n"

            supporting_info_e += "related triples:" + ", ".join(map(str, self.ent2triples.get(entity, []))) + "\n"
            supporting_info_e += "related paths:" + self.ent2paths.get(entity, "") + ".\n"
        for relation, text in self.rel2text.items():
            supporting_info_r += relation + ":" + text + "\n"
            supporting_info_r += "corresponding description:" + self.rel2align_text.get(relation, "") + "\n"
            supporting_info_r += "related triples:" + ", ".join(map(str, self.rel2triples.get(relation, [])) )+ "\n"
        with open(self.args.supporting_file_path, 'w') as f:
            f.write(supporting_info_e)
            f.write(supporting_info_r)
        print(f"The supporting information of entity & relation has been saved to {self.args.supporting_file_path}")

    def data_combine_dict(self):
        supporting_info_e = {}
        supporting_info_r = {}

        for entity, text in self.ent2text.items():
            supporting_info_e[entity] = {
                "entity": entity,
                "name": text,
                "triples": self.ent2triples.get(entity, []),
                "description": self.entity2description_clean.get(entity, ""),
                "paths": self.ent2paths.get(entity, "")

            }

        for relation, text in self.rel2text.items():
            supporting_info_r[relation] = {
                "relation": relation,
                "name": text,
                "description": self.rel2align_text.get(relation, ""),
                "triples": self.rel2triples.get(relation, [])
            }
        with open(self.args.supporting_file_path, 'w') as f:
            json.dump(supporting_info_e, f, ensure_ascii=False, indent=2)
            json.dump(supporting_info_r, f, ensure_ascii=False, indent=2)

        print(f"The supporting information of entity & relation has been saved to {self.args.supporting_file_path}")

def main(args, entities, idx, api_key):
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

    parser.add_argument('--api_key', default="Your API", type=str)
    parser.add_argument('--num_process', default=1, type=int, help='the number of multi-process')


    parser.add_argument('--max_tokens', default=500, type=int, help='max-token')
    args = parser.parse_args()

    print("Start querying the LLM.")
    return args
def run_worker(idx, args, sub_entities, api_key):
    main(args, sub_entities, idx, api_key)
def merge_outputs(args):
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

    print(f"Merging Finished：{final_output}, {final_chatlog}")


if __name__ == '__main__':
    args = parse_args()
    if not args.api_key.startswith("sk-"):
        with open(args.api_key, "r") as f:
            all_keys = f.readlines()
            all_keys = [line.strip('\n') for line in all_keys]
            assert len(all_keys) == args.num_process, (len(all_keys), args.num_process)
        
    ent2text = defaultdict(str)
    with open('dataset/' + args.dataset + '/entity2text.txt', 'r') as file:
        entity_lines = file.readlines()
        for line in entity_lines:
            ent, text = line.strip().split("\t")
            ent2text[ent] = text


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

        merge_outputs(args)

    else:
        main(args, ent2text, idx=-1, api_key=args.api_key)








