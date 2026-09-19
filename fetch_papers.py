import arxiv
import os
import requests

def fetch_papers(query='retrieval augmented generation', max_results=5, out_dir='data'):
    os.makedirs(out_dir, exist_ok=True)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    fetched = []
    for result in client.results(search):
        safe_id = result.entry_id.split('/')[-1]
        filepath = os.path.join(out_dir, f"{safe_id}.pdf")
        pdf_url = result.pdf_url
        response = requests.get(pdf_url)
        with open(filepath, 'wb') as f:
            f.write(response.content)
        fetched.append({
            'id': safe_id,
            'title': result.title,
            'path': filepath
        })
        print(f'Downloaded: {result.title}')
    
    return fetched 

if __name__ == '__main__':
    fetch_papers()