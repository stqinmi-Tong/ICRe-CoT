'''
Prompt Engineering: 1. Add comparable  2. Add confidence scoreto better and confident to be best. 3. Final Have a try again 4. rethinking 5.vote 6.repeat
7.iterative or accumulated update  8.question is/are
'''
import json
from gensim.summarization.bm25 import BM25
from gensim import corpora
import operator
from collections import defaultdict
import random
import argparse
import heapq
from sentence_transformers import SentenceTransformer
class Demon_sampler:
    def __init__(self, args):
        self.ent2text = defaultdict(str)
        self.entity_supplement = defaultdict(list)
        self.relation_analogy = defaultdict(list)
        self.T_link_base = defaultdict(list)
        self.link_base = defaultdict(list) #t+/t+r:[[h1,r,t],[h2,r,t],...] or h+/t+r:[[h,r,t1],[h,r,t2],...]
        self.link_base_txt = defaultdict(list)
        self.args = args
        self.dataset = args.dataset
        self.load_ent_to_text()
        self.load_demonstration()
        # self.shrink_link_base()
        # self.demo_list_execution = []
    def load_ent_to_text(self):
            with open('/root/autodl-tmp/dataset/' + self.dataset + '/entity2text.txt', 'r') as file:
                entity_lines = file.readlines()
                for line in entity_lines:
                    ent, text = line.strip().split("\t")
                    self.ent2text[ent] = text
    def load_demonstration(self):
        with open("/root/autodl-tmp/dataset/" + self.dataset + "/demonstration/"+ "T_link_base_"+ self.args.query +".txt", "r") as f:
            self.link_base = json.load(f)  ###t+/t+r:[[h1,r,t],[h2,r,t],...] or h+/t+r:[[h,r,t1],[h,r,t2],...]

        with open("/root/autodl-tmp/dataset/" + self.dataset + "/demonstration/"+ self.args.query +"_supplement.txt", "r") as f:
            supplement_pool = json.load(f)

        with open("/root/autodl-tmp/dataset/" + self.dataset + "/demonstration/"+ self.args.query +"_analogy.txt", "r") as f:
            analogy_pool = json.load(f)

        keys = self.ent2text.keys()
        for key in supplement_pool:
            tmp_list = []
            for value in supplement_pool[key]:
                if value[0] in keys:
                    tmp_list.append([self.ent2text[value[0]],value[1],self.ent2text[value[2]]])
                else:
                    tmp_list.append([value[0],value[1],value[2]])

            self.entity_supplement[key] = tmp_list
        for key in analogy_pool:
            tmp_list = []
            for value in analogy_pool[key]:
                if value[0] in keys:
                    tmp_list.append([self.ent2text[value[0]],value[1],self.ent2text[value[2]]])
                else:
                    tmp_list.append([value[0],value[1],value[2]])
                # random.shuffle(tmp_list)
            self.relation_analogy[key] = tmp_list
        for key in self.link_base: #self.link_base: t+/t+r:[[h1,r,t],[h2,r,t],...] or h+/t+r:[[h,r,t1],[h,r,t2],...]
            tmp_list = []
            for value in self.link_base[key]:
                if value[0] in keys:
                    tmp_list.append([self.ent2text[value[0]],value[1],self.ent2text[value[2]]])
                else:
                    tmp_list.append([value[0],value[1],value[2]])
                
            self.link_base_txt[key] = tmp_list

            
    def true_candidates(self,h,r): #self.T_link_base[h+/t+r]=[h_text,r,'t1_text,t2_text,...']
        return self.T_link_base['\t'.join([h,r])][2] #得到h+/t+r对应的t列表的字符串

    def true_candidate_v2(self, h, r, num):
        ##self.link_base_txt： t+/t+r:[[h1_text,r,t_text],[h2_text,r,t_text],...]
        # or h+/t+r:[[h_text,r,t1_text],[h_text,r,t2_text],...]
        ####存疑啊，这个link_base_txt取自self.link_base，到底包不包含test测试集中的三元组呢？？存疑啊
        true_set = self.link_base_txt['\t'.join([h, r])][:num]
        return true_set  ###[[h_text,r,t1_text],[h_text,r,t2_text],...]
          
    def Diversity_arranged(self, tpe, relation):
        ##该函数的目的是：对给定 tpe 和 relation 对应的三元组列表 demon_list 进行重排序，
        # 使得最终排序后的三元组列表中实体尽可能多样化（即尽量避免相同的实体重复出现）
        ''' 应用场景（举例）：在构造知识图谱的类比推理训练样本时：

          不希望某个实体如 "Germany" 或 "Barack Obama" 在多条样本中频繁出现；

          这个函数可以自动排序，使不同实体分布更均衡，从而提升训练或评估的泛化能力。
        '''
        demon_list = self.relation_analogy['\t'.join([tpe, relation])]
        entity_counter = defaultdict(int)
        def count_sum(triple):
            return entity_counter[triple[0]] + entity_counter[triple[2]], triple
        priority_queue = [count_sum(triple) for triple in demon_list]
        heapq.heapify(priority_queue)

        sorted_list = []
        while priority_queue:
            _, next_triple = heapq.heappop(priority_queue)
            sorted_list.append(next_triple)
            entity_counter[next_triple[0]] += 1
            entity_counter[next_triple[2]] += 1
            priority_queue = [count_sum(triple) for triple in priority_queue]
            heapq.heapify(priority_queue)
        self.relation_analogy['\t'.join([tpe, relation])] = sorted_list

    def Diversity_arranged_v2(self, tpe, entity2triples):
        ##该函数的目的是：对给定实体tpe对应的三元组列表demon_list进行重排序，
        # 使得最终排序后的三元组列表中实体尽可能多样化（即尽量避免相同的实体重复出现）
        ''' 应用场景（举例）：在构造知识图谱的类比推理训练样本时：

          不希望某个实体如 "Germany" 或 "Barack Obama" 在多条样本中频繁出现；

          这个函数可以自动排序，使不同实体分布更均衡，从而提升训练或评估的泛化能力。
        '''
        demon_list = entity2triples[tpe]
        entity_counter = defaultdict(int)
        def count_sum(triple):
            return entity_counter[triple[0]] + entity_counter[triple[2]], triple
        priority_queue = [count_sum(triple) for triple in demon_list]
        heapq.heapify(priority_queue)

        sorted_list = []
        while priority_queue:
            _, next_triple = heapq.heappop(priority_queue)
            sorted_list.append(next_triple)
            entity_counter[next_triple[0]] += 1
            entity_counter[next_triple[2]] += 1
            priority_queue = [count_sum(triple) for triple in priority_queue]
            heapq.heapify(priority_queue)
        return sorted_list

        
    def BM25_arranged(self, tpe, relation):
        ##使用 BM25 算法根据查询语句对一组候选三元组进行相关性打分与排序，提升与查询最相关的三元组的优先级。
        #该函数基于 BM25 信息检索算法，对候选知识图谱三元组列表进行语义相关性排序，使其更贴近给定实体和关系构成的查询。
        """
        应用场景：通常用于知识图谱问答或链接预测任务中的 示例排序；能够提升示例选择的相关性；
        可用于训练集中“支持三元组”的排序优化，以提升模型感知上下文能力。

        """
        demon_list = self.entity_supplement['\t'.join([tpe, relation])]
        tpe_text = self.ent2text[tpe]
        question_text = tpe_text + relation if self.args.query == 'tail' else relation + tpe_text
        texts = ['\t'.join(triple) for triple in demon_list]
        dictionary = corpora.Dictionary([text.split() for text in texts])
        corpus = [dictionary.doc2bow(text.split()) for text in texts]
        bm25 = BM25(corpus)
        query = dictionary.doc2bow(question_text.split())
        scores = bm25.get_scores(query) #用构造的 query 向量去 BM25 模型中获取与所有候选三元组的相似度打分；
        scored_triples = list(zip(demon_list, scores))
        sorted_triples = sorted(scored_triples, key=lambda x: x[1], reverse=True)
        sorted_demon_list = [triple for triple, score in sorted_triples]
        self.entity_supplement['\t'.join([tpe, relation])] = sorted_demon_list

        
    def poolsampler(self, tpe, r, num, step_num):
        ##为给定实体类型 tpe 和关系 r 从两个三元组池
        # （类比池 relation_analogy 和补充池 entity_supplement）中分批采样固定数量的示例三元组，
        # 并返回本轮采样的子集，用于构造训练样本或推理支持集。
        analogy_num = num//2
        supplement_num = num - analogy_num
        start_analogy = step_num * analogy_num
        end_analogy = start_analogy + analogy_num
        start_supple = step_num * supplement_num
        end_supple = start_supple + supplement_num
        if '\t'.join([tpe, r]) not in self.demo_list_execution:
            self.Diversity_arranged(tpe, r)
            self.BM25_arranged(tpe, r)
            self.demo_list_execution.append('\t'.join([tpe, r]))
        analogy_arranged_set = self.relation_analogy['\t'.join([tpe, r])]
        supplement_arranged_set = self.entity_supplement['\t'.join([tpe, r])]
        analogy_set = analogy_arranged_set[start_analogy:end_analogy]
        supplement_set = supplement_arranged_set[start_supple:end_supple]
        return analogy_set,supplement_set

    def poolsampler_v2(self, tpe, entity2triples,num):
        ##为给定实体类型 tpe 和关系 r 从两个三元组池
        # entity2triples中分批采样固定数量的示例三元组，
        # 并返回本轮采样的子集，用于构造训练样本或推理支持集。
        sorted_list = self.Diversity_arranged_v2(tpe, entity2triples)
        selected_list = sorted_list[:num]
        return selected_list

    def sample_paths_diverse(self, paths, k, strategy):
        """
        paths: List[List[str]] - 多跳路径
        k: int - 要采样的数量
        strategy: str - 采样策略
            one of ["entity_diversity", "entity_frequency", "relation_diversity", "embedding_clustering"]
        embeddings: np.ndarray - 路径的嵌入（用于 embedding_clustering）
        """

        if strategy == "entity_diversity":
            return sample_entity_diversity(paths, k)

        elif strategy == "entity_frequency":
            return sample_entity_frequency(paths, k, max_entity_freq=3)

        elif strategy == "relation_diversity":
            return sample_relation_diversity(paths, k)

        elif strategy == "embedding_clustering":
            if embeddings is None:
                raise ValueError("Embeddings required for 'embedding_clustering' strategy.")
            return sample_embedding_clustering(paths, k, 6)

        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def sample_entity_diversity(self, paths, k):
        selected_paths = []
        used_entities = set()

        for path in paths:
            entities_in_path = set(path[::2][1:])  # 所有实体
            if len(entities_in_path & used_entities) == 0:
                selected_paths.append(path)
                used_entities.update(entities_in_path)
            elif len(selected_paths) < k // 2:
                selected_paths.append(path)
                used_entities.update(entities_in_path)
            if len(selected_paths) >= k:
                break

        # 不足则补齐
        if len(selected_paths) < k:
            extra = [p for p in paths if p not in selected_paths]
            selected_paths.extend(random.sample(extra, min(k - len(selected_paths), len(extra))))
        return selected_paths

    def sample_entity_frequency(self, paths, k, max_entity_freq=2):
        selected = []
        entity_count = defaultdict(int)

        for path in paths:
            entities = path[::2][1:]  # 除起始实体外的实体
            if all(entity_count[e] < max_entity_freq for e in entities):
                selected.append(path)
                for e in entities:
                    entity_count[e] += 1
            if len(selected) >= k:
                break

        if len(selected) < k:
            extra = [p for p in paths if p not in selected]
            selected.extend(random.sample(extra, min(k - len(selected), len(extra))))
        return selected

    def sample_relation_diversity(self, paths, k):
        selected = []
        used_patterns = set()

        for path in paths:
            relations = tuple(path[1::2])
            if relations not in used_patterns:
                selected.append(path)
                used_patterns.add(relations)
            if len(selected) >= k:
                break

        if len(selected) < k:
            extra = [p for p in paths if p not in selected]
            selected.extend(random.sample(extra, min(k - len(selected), len(extra))))
        return selected

    def sample_embedding_clustering(self, paths, k, n_clusters=10):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [" -> ".join(p) for p in paths]
        embeddings = model.encode(texts)
        if len(paths) <= k:
            return paths

        kmeans = KMeans(n_clusters=min(n_clusters, len(paths)), random_state=42)
        labels = kmeans.fit_predict(embeddings)

        cluster_to_paths = defaultdict(list)
        for path, label in zip(paths, labels):
            cluster_to_paths[label].append(path)

        selected = []
        for cluster_paths in cluster_to_paths.values():
            selected.append(random.choice(cluster_paths))
            if len(selected) >= k:
                break
        if len(selected) < k:
            extra = [p for p in paths if p not in selected]
            selected.extend(random.sample(extra, min(k - len(selected), len(extra))))

        return selected


          
    def randomsampler(self, tpe, r, num, step_num): # need a new version for no repeat facts
        analogy_num = num//2
        supplement_num = num - analogy_num
        start_analogy = step_num * analogy_num
        end_analogy = start_analogy + analogy_num
        start_supple = step_num * supplement_num
        end_supple = start_supple + supplement_num
        analogy_set = self.relation_analogy['\t'.join([tpe, r])][start_analogy:end_analogy]
        supplement_set = self.entity_supplement['\t'.join([tpe, r])][start_supple:end_supple]
        return analogy_set, supplement_set ###每个step取num//2的analogy例子和num-analogy_num的supplement例子

        
    def shrink_link_base(self):
        ##从文件加载三元组链路基础（link base），对每个链路样本进行缩减处理，
        # 仅保留前 10 个目标实体的信息，并构造成简洁的 [头实体文本，关系名，若干目标实体文本] 结构，
        # 存入 self.T_link_base 中，作为简化后的链路知识库。

        ###感觉这只是处理了self.args.query=tail_prediction的情况，得到self.T_link_base[h+/t+r]=[h_text,r,'t1_text,t2_text,...']
        for key in self.link_base:
            if len(self.link_base[key]) == 0: 
                self.T_link_base[key] = []
                break
            h,r = key.split('\t')
            enetity_link_base = ""
            for value in self.link_base[key][:10]:
                h_text = self.ent2text[value[0]]
                enetity_link_base += self.ent2text[value[2]] + ','
            enetity_link_base.strip(',')
            # if enetity_link_base == "": enetity_link_base = "None"
            self.T_link_base[key] = [h_text,r,enetity_link_base]






# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, default=None)
#     args = parser.parse_args()
#     data_sampler = Demon_sampler(args)
#     maxlen = 0
#     maxkey = ''
#     for key in data_sampler.T_link_base:
#         print(len(data_sampler.T_link_base[key][2]))
#         if len(data_sampler.T_link_base[key][2]) > maxlen:
#             maxlen = len(data_sampler.T_link_base[key][2])
#             maxkey = key
#     print(maxlen,maxkey)