from retriever import search_docs
from sklearn.metrics.pairwise import cosine_similarity


def get_content(question, model, index):
    matches =  search_docs(question, model, index)
    contexts = []
    if matches:
        for match in matches:
            if "metadata" in match and "text" in match["metadata"]:
                contexts.append(match['metadata']['text'])
            else:
                print(f"Unexpected match format {match}")
    else:
        print("No matches found")

    context = "".join(contexts)
    return context
