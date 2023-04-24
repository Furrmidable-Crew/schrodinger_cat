import time

from langchain.chains.qa_with_sources import load_qa_with_sources_chain
import threading
from llama_index import download_loader
from pymed import PubMed
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain.prompts import PromptTemplate
from cat.utils import log
from cat.mad_hatter.decorators import tool, hook
from langchain.docstore.document import Document
from langchain.output_parsers import RegexParser
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.chains.summarize import load_summarize_chain


class SchrodingerCat(PubMed):

    def __init__(self, ccat):
        # Download the loader
        pubmedreader = download_loader("PubmedReader")

        # Loader
        self.loader = pubmedreader()

        # Cheshire Cat
        self.cat = ccat

    @staticmethod
    def parse_query(tool_input):
        # Split the inputs
        multi_input = tool_input.split(",")
        log(multi_input)
        # TODO: check max results is an integer as sometimes the Cat leave a quote
        # e.g. multi_input[1]= "1'" that can't be cast to int
        if len(multi_input) == 1:
            max_results = 1
            query = multi_input
        else:
            max_results = int(multi_input[1])
            query = multi_input[0]

        return query, max_results

    @staticmethod
    def __postprocess_docs(docs):
        # TODO: review how to define source
        for doc in docs:
            if "Author" in doc.metadata.keys():
                author = doc.metadata["author"]
            if "Title of this paper" in doc.metadata.keys():
                title = doc.metadata["Title of this paper"]
            if "Journal it was published in" in doc.metadata.keys():
                journal = doc.metadata["Journal it was published in"]
            if "URL" in doc.metadata.keys():
                url = doc.metadata["URL"]
            source = f"{title}"
            doc.metadata["source"] = source
        return docs

    def __query(self, query: str, max_results: int = 1):
        # Get Documents from query
        docs = self.loader.load_data(search_query=query, max_results=max_results)

        # Log info - to be deleted
        log(f" Retrieved {len(docs)}. Converting to langchain documents")

        # Convert to Langchain Documents
        langchain_documents = [d.to_langchain_format() for d in docs]

        # Postprocess retrieved docs
        documents = self.__postprocess_docs(langchain_documents)

        # Store docs in Working Memory to further operations.
        # e.g. filter docs
        self.cat.working_memory.keep_in_mind(documents)

        # Log info - to be deleted
        log(self.cat.working_memory.memories[0].metadata["source"])

    def make_search(self, tool_input):
        # Split input in str and int
        query, max_results = self.parse_query(tool_input)

        # Make concurrent task to download paper in background if max_results is high
        search = threading.Thread(target=self.__query, name="PubMed query", args=[query, max_results])
        search.start()


@tool(return_direct=True)
def simple_search(query: str, cat):
    """
    Useful to look for a query on PubMed. It is possible to specify the number of results desired.
    The input to this tool should be a comma separated list of a string and an integer number.
    The integer number is optional and if not provided is set to 1.
    For example: 'Antibiotic,5' would be the input if you want to look for 'Antibiotic' with max 5 results.
    Another example: 'Antibiotic,1' would be the input if only the query 'Antibiotic' is asked.
    To use this tool start the whole prompt with PUBMED: written in uppercase.
    Examples:
         - PUBMED: Look for "Public Healthcare" and give me 3 results.' Input is 'Public Healthcare,3'
         - PUBMED: Look for "Antibiotic resistance".' Input is 'Public Healthcare,1'
    """

    # Schrodinger Cat
    schrodinger_cat = SchrodingerCat(cat)

    # Search on PubMed
    schrodinger_cat.make_search(query)

    # TODO: change this output
    out = f"Alright. I'm looking for {schrodinger_cat.parse_query(query)[1]} results about" \
          f" {schrodinger_cat.parse_query(query)[0]} on PubMed. This may take some time. " \
          f"Hang on please, I'll tell you when I'm done"

    return out


@tool(return_direct=True)
def empty_working_memory(tool_input, cat):
    """
    Useful to empty and forget all the documents in the Working Memory. Input is always None.
    """

    # Empty working memory
    cat.working_memory.forget()

    # TODO: this has to be tested
    # the idea is having the Cat answer without directly returning a hard coded output string
    return cat.llm("Can you please forget everything I asked you to keep in mind?")


@tool()
def query_working_memory(tool_input, cat):
    """
    Useful to ask for reviewing a paper. The paper is a document in the Working Memory. Input is always None.
    """
    # TODO return detailed info about what's in the Working Memory
    # Notes: as llama_index doesn't give much details about the papers,
    # we can use the titles to query individual papers with pymed and retrieved all the info
    # (authors, affiliation, citations?, journal, url, etc.,
    # then format the output expecting user to ask for more info about a specific paper
    return "ciao"


@tool
def explain_paper(tool_input, cat):
    """
    Useful to have a paper explained in a systematic way.
    The Cat answers 10 questions from
    'Carey MA, Steiner KL, Petri WA Jr. Ten simple rules for reading a scientific paper'
    Input to this tool is the title of a paper.
    """

    # Set paper to none
    paper = None

    # Look for the Title of the paper in Working Memory
    for m in cat.working_memory.memories:
        if m.metadata['Title of this paper'] == tool_input:
            paper = m

    # If not in Worming Memory, search in Long Term Memory
    if paper is None:
        paper = cat.memory.vectors.declarative.similarity_search(
            query=tool_input,
            k=1,
            filter={"Title of this paper": tool_input}
        )

    # If not in Long Term Memory return telling it doesn't exist
    if paper is None:
        return "I don't have the paper you queried for"

    # Text splitter (big chunks good idea?)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=1500,
                                                   separators=["\\n\\n", "\n\n", ".\\n", ".\n", "\\n", "\n", " ", ""])

    # Split the document
    docs = text_splitter.split_documents([paper])

    # Log info - to be deleted
    log(len(docs))

    # docs = list(filter(lambda d: len(d.page_content) > 10, docs))

    # Questions from 'Carey MA, Steiner KL, Petri WA Jr. Ten simple rules for reading a scientific paper'
    questions = [
        'What do the authors want to know? What is the motivation?',
        'What did they do? What is the approach/methods?',
        'Why was it done that way? Which is the context within the field of research?',
        'What do the results show?',
        'How did the authors interpret the results? Which is their discussion?',
        'What should be done next? The authors may provide some suggestions in the discussion'
    ]

    # Summarize with refine because most lossless chain
    chain = load_summarize_chain(cat.llm, chain_type="refine")

    # Summarize
    summaries = [chain.run([d]) for d in docs]

    # Log info - to be deleted
    log(summaries)

    # Make document from summaries - metadata?
    s = [Document(page_content=ss) for ss in summaries]

    # Question & Answer chain
    # TODO two refine takes ages, test stuff in q&a
    qa_chain = load_qa_chain(cat.llm, chain_type="refine")  # cambia chain type

    # Gather answers
    answers = [qa_chain({"input_documents": s, "question": q}, return_only_outputs=True) for q in questions]
    log(answers)

