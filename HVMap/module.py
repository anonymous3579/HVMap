class HVMap:
    def __init__(
        self,
        model_from,
        model_to,
        dataset_name,
        embedding_model_from,
        embedding_model_to,
        index_from,
        index_to,
        converter,
        converter2,
        q_text,
        k=10,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_from = model_from
        self.model_to = model_to
        self.dataset_name = dataset_name
        self.k = k
        self.q_text = q_text  # externally provided

        self.index_from = index_from
        self.index_from.set_query_arguments(128)

        self.index_to = index_to
        self.index_to.set_query_arguments(128)

        self.embedding_model_from = embedding_model_from
        self.embedding_model_to = embedding_model_to

        self.converter = converter.to(self.device)
        self.converter2 = converter2.to(self.device)

    def query(self, converting_parm=0.75):
        import torch
        import torch.nn.functional as F
        import numpy as np

        from_query = self.embedding_model_from.encode_queries(self.q_text)
        to_query = self.converter(from_query)

        res_from = [self.index_from.query(from_query[i], self.k) for i in range(len(self.q_text))]
        res_to = [self.index_to.query(to_query[i], self.k) for i in range(len(self.q_text))]

        # From side
        all_vectors, all_fq, all_tq = [], [], []
        for i, res in enumerate(res_from):
            vectors = torch.from_numpy(res[1])
            fq = torch.from_numpy(from_query[i]).unsqueeze(0).repeat(len(res[0]), 1)
            tq = torch.from_numpy(to_query[i]).unsqueeze(0).repeat(len(res[0]), 1)

            all_vectors.append(vectors)
            all_fq.append(fq)
            all_tq.append(tq)

        all_vectors = torch.cat(all_vectors).to(self.device)
        all_fq = torch.cat(all_fq).to(self.device)
        all_tq = torch.cat(all_tq).to(self.device)

        converted_vectors = self.converter(all_vectors)
        dist1 = F.cosine_similarity(all_vectors, all_fq, dim=-1)
        dist2 = F.cosine_similarity(converted_vectors, all_tq, dim=-1)
        final_dist1 = (1 - converting_parm) * dist2 + converting_parm * dist1

        res_from_dic, pointer = [], 0
        for res in res_from:
            k = len(res[0])
            res_from_dic.append({
                "doc-id": res[0],
                "distance": final_dist1[pointer:pointer + k]
            })
            pointer += k

        # To side
        all_vectors, all_fq, all_tq = [], [], []
        for i, res in enumerate(res_to):
            vectors = torch.from_numpy(res[1])
            fq = torch.from_numpy(from_query[i]).unsqueeze(0).repeat(len(res[0]), 1)
            tq = torch.from_numpy(to_query[i]).unsqueeze(0).repeat(len(res[0]), 1)

            all_vectors.append(vectors)
            all_fq.append(fq)
            all_tq.append(tq)

        all_vectors = torch.cat(all_vectors).to(self.device)
        all_fq = torch.cat(all_fq).to(self.device)
        all_tq = torch.cat(all_tq).to(self.device)

        converted_vectors = self.converter2(all_vectors)
        dist1 = F.cosine_similarity(all_vectors, all_tq, dim=-1)
        dist2 = F.cosine_similarity(converted_vectors, all_fq, dim=-1)
        final_dist2 = (1 - converting_parm) * dist2 + converting_parm * dist1

        res_to_dic, pointer = [], 0
        for res in res_to:
            k = len(res[0])
            res_to_dic.append({
                "doc-id": res[0],
                "distance": final_dist2[pointer:pointer + k]
            })
            pointer += k

        # Merge ranking
        sort_convert = []
        for r1, r2 in zip(res_from_dic, res_to_dic):
            combined_dist = torch.cat([r1["distance"], r2["distance"]])
            combined_doc_ids = np.concatenate([r1["doc-id"], r2["doc-id"]])
            sorted_idx = torch.argsort(-combined_dist)
            sort_convert.append(combined_doc_ids[sorted_idx.cpu()].tolist())

        return sort_convert
