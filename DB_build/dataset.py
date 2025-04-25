# utils/IR_task.py
from datasets import Dataset
from typing import Callable

class __DatasetBuilder:
    def __init__(self, parsers: list[Callable]):
        self.__parsers = parsers

    def build(self, dataset):
        for parser in self.__parsers:
            dataset = parser(dataset)
        return dataset 

def __split_dataset(dataset):
    import pandas, re
    new_dataset = []
    for row in dataset:
        relevant_ids = re.split(r',\s*', row['relevant_passage_ids'].strip("[]"))
        for rel_id in relevant_ids:
            new_dataset.append({
                'query-id': row['id'],
                'doc-id': int(rel_id),  # rel_id를 정수로 변환
            })
    new_dataset = pandas.DataFrame(new_dataset)
    new_dataset = Dataset.from_pandas(new_dataset)
    return new_dataset

def __add_prefix(dataset, column:str="doc-id", prefix:str="prefix"):
    def callback(row):
        row[column] = f"{prefix}-{row[column]}"
        return row
    return dataset.map(callback)

DATASET_BUILDERS: dict[str, __DatasetBuilder] = {
    # masmarco
    "BeIR/msmarco/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/msmarco/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/msmarco-qrels/train": __DatasetBuilder([
        lambda dataset : dataset["train"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # cqadupstack
    "BeIR/cqadupstack-generated-queries": __DatasetBuilder([
        lambda dataset : dataset["train"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/cqadupstack-qrels": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # NFCorpus is a full-text English retrieval data set for Medical Information Retrieval
    "BeIR/nfcorpus/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/nfcorpus/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/nfcorpus-qrels/train": __DatasetBuilder([
        lambda dataset : dataset["train"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    "BeIR/nfcorpus-qrels/validation": __DatasetBuilder([
        lambda dataset : dataset["validation"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    "BeIR/nfcorpus-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # hotpotqa : question-answering system reads 10 paragraphs to provide an answer (Ans) to a question
    "BeIR/hotpotqa/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/hotpotqa/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/hotpotqa-qrels/train": __DatasetBuilder([
        lambda dataset : dataset["train"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    "BeIR/hotpotqa-qrels/validation": __DatasetBuilder([
        lambda dataset : dataset["validation"],
        lambda dataset : dataset.remove_columns(["score"]),
    ]),
    "BeIR/hotpotqa-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # nq : 
    "BeIR/nq/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/nq/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/nq-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # FiQA : Financial Opinion Mining and Question Answering
    "BeIR/fiqa/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "fin"),
    ]),
    "BeIR/fiqa/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/fiqa-qrels/train": __DatasetBuilder([
        lambda dataset : dataset["train"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "fin"),
    ]),
    "BeIR/fiqa-qrels/validation": __DatasetBuilder([
        lambda dataset : dataset["validation"],
        lambda dataset : dataset.remove_columns(["score"]),
    ]),
    "BeIR/fiqa-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "fin"),
    ]),
    # arguana : 
    "BeIR/arguana/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/arguana/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/arguana-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # Touche-2020 : Given a question on a controversial topic, retrieve relevant arguments from a focused crawl of online debate portals
    "BeIR/webis-touche2020/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/webis-touche2020/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/webis-touche2020-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # Quora : 
    "BeIR/quora/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/quora/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/quora-qrels/validation": __DatasetBuilder([
        lambda dataset : dataset["validation"],
        lambda dataset : dataset.remove_columns(["score"]),
    ]),
    "BeIR/quora-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
    ]),
    # DBPedia : standard test collection for entity search over the DBpedia knowledge base
    "BeIR/dbpedia-entity/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/dbpedia-entity/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/dbpedia-entity-qrels/validation": __DatasetBuilder([
        lambda dataset : dataset["validation"],
        lambda dataset : dataset.remove_columns(["score"]),
    ]),
    "BeIR/dbpedia-entity-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["validation"],
        lambda dataset : dataset.filter(lambda row: row['score'] >= 1),
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # SCIDOCS : Dataset Evaluation Suite for SPECTER
    "BeIR/scidocs/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "sci"),
    ]),
    "BeIR/scidocs/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/scidocs-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.filter(lambda row: row['score'] == 1),
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "sci"),
    ]),
    # FEVER : 
    "BeIR/fever/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/fever/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/fever-qrels/train": __DatasetBuilder([
        lambda dataset : dataset["train"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    "BeIR/fever-qrels/validation": __DatasetBuilder([
        lambda dataset : dataset["validation"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    "BeIR/fever-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # Climate-fever : CLIMATE-FEVER is a dataset adopting the FEVER methodology that consists of 1,535 real-world claims regarding climate-change
    "BeIR/climate-fever/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/climate-fever/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/climate-fever-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
    ]),
    # scifact : 
    "BeIR/scifact/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "sci"),
    ]),
    "BeIR/scifact/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/scifact-qrels/train": __DatasetBuilder([
        lambda dataset : dataset["train"],
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "sci"),
    ]),
    "BeIR/scifact-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["score"]),
    ]),
    # TREC-COVID : unique opportunity for the information retrieval (IR) and text processing communities to contribute to the response to this pandemic
    "BeIR/trec-covid/corpus": __DatasetBuilder([
        lambda dataset : dataset["corpus"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "doc-id"),
    ]),
    "BeIR/trec-covid/queries": __DatasetBuilder([
        lambda dataset : dataset["queries"],
        lambda dataset : dataset.remove_columns(["title"]),
        lambda dataset : dataset.rename_column("_id", "query-id"),
    ]),
    "BeIR/trec-covid-qrels/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.filter(lambda row: row['score'] >= 1),
        lambda dataset : dataset.remove_columns(["score"]),
        lambda dataset : dataset.rename_column("corpus-id", "doc-id"),
    ]),
    # rag-mini-bioasq
    "rag-datasets/rag-mini-bioasq/corpus": __DatasetBuilder([
        lambda dataset : dataset["passages"],
        lambda dataset : dataset.filter(lambda row: row["passage"] != "nan"),
        lambda dataset : dataset.rename_column("passage", "text"),
        lambda dataset : dataset.rename_column("id", "doc-id"),
        lambda dataset : __add_prefix(dataset, "doc-id", "bio"),
    ]),
    "rag-datasets/rag-mini-bioasq/queries": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : dataset.remove_columns(["answer", "relevant_passage_ids"]),
        lambda dataset : dataset.rename_column("question", "text"),
        lambda dataset : dataset.rename_column("id", "query-id"),
    ]),
    "rag-datasets/rag-mini-bioasq/test": __DatasetBuilder([
        lambda dataset : dataset["test"],
        lambda dataset : __split_dataset(dataset),
    ]),
}

# TEST CODE
if __name__ == "__main__":
    from datasets import load_dataset

    # # Example1 : Usage for corpus
    # corpus = DATASET_BUILDERS["BeIR/msmarco/corpus"].build(load_dataset("BeIR/msmarco", 'corpus'))
    # print(corpus[0]['text'])  # [{'doc-id':'~', 'text':'~'}]

    # # Example2 : Usage for queries
    # query = DATASET_BUILDERS["BeIR/msmarco/queries"].build(load_dataset("BeIR/msmarco", "queries"))
    # print(query[0])   # [{'query-id':'~', 'text':'~'}]
    
    # # Example3 : Usage for qrels
    # qrels = DATASET_BUILDERS["BeIR/msmarco-qrels/train"].build(load_dataset("BeIR/msmarco-qrels", "default"))
    # print(qrels[0])   # [{'query-id':'~', 'doc-id':'~'}]

    # # Example4 : Usage for filtering qrels
    # doc_ids_for_query_id_8 = [row["doc-id"] for row in qrels.filter(lambda row : row["query-id"] == 1185869)]
    # print(doc_ids_for_query_id_8)   # [566392, 65404]
    
    # print("====")
    
    # corpus = DATASET_BUILDERS["BeIR/trec-covid/corpus"].build(load_dataset("BeIR/trec-covid", "corpus"))
    # print(corpus[0])
    # qrels = DATASET_BUILDERS["BeIR/trec-covid-qrels/test"].build(load_dataset("BeIR/trec-covid-qrels"))
    # print(qrels[0])
    # qrels = DATASET_BUILDERS["BeIR/scidocs-qrels/test"].build(load_dataset("BeIR/scidocs-qrels"))
    # print(qrels[0])

    ############
    # 24.12.18 #
    ############
    # corpus = DATASET_BUILDERS["rag-datasets/rag-mini-bioasq/corpus"].build(load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus"))
    # print(corpus[0])

    # queries = DATASET_BUILDERS["rag-datasets/rag-mini-bioasq/queries"].build(load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages"))
    # print(queries[0])

    # qrels = DATASET_BUILDERS["rag-datasets/rag-mini-bioasq/test"].build(load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages"))
    # print(qrels[0])

    # doc_ids_for_query_id_0 = [row["doc-id"] for row in qrels.filter(lambda row : row["query-id"] == 0)]
    # print(doc_ids_for_query_id_0)

    scifact_train = DATASET_BUILDERS["BeIR/fiqa/corpus"].build(load_dataset("BeIR/fiqa", "corpus"))
    print(scifact_train[0])