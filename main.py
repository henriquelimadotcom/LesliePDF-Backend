import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',
                        type=str,
                        default='How much is the minimum guarantee payable by adidas?',
                        help='Enter the query to pass into the LLM')
    args = parser.parse_args()

    # Setup DBQA
    start = timeit.default_timer()
    dbqa = setup_dbqa()
    response = dbqa({'query': args.input})
    end = timeit.default_timer()

    print(f'\nAnswer: {response["result"]}')
    print('='*50)

    # Process source documents
    source_docs = response['source_documents']
    for i, doc in enumerate(source_docs):
        print(f'\nSource Document {i+1}\n')
        print(f'Source Text: {doc.page_content}')

        # Create a clickable link to the PDF
        file_path = doc.metadata["source"]
        page_number = doc.metadata["page"]
        link = f"file://{file_path}#page={page_number}"
        
        print(f'Document Name: \033]8;;{link}\033\\{file_path}\033]8;;\033\\')
        print(f'Page Number: {page_number}\n')
        print('='* 60)
        
    print(f"Time to retrieve response: {end - start}")
