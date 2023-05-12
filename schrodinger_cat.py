import threading
from pymed import PubMed
from langchain.text_splitter import RecursiveCharacterTextSplitter
from cat.utils import log
from cat.mad_hatter.decorators import tool, hook
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain

from transformers import AutoTokenizer, OPTForCausalLM


class SchrodingerCat:

    def __init__(self, ccat):

        # PyMed
        self.pymed = PubMed(tool="mytool", email="myemail@email.com")

        # Cheshire Cat
        self.cat = ccat

    @staticmethod
    def parse_query(tool_input):

        # Split the inputs
        multi_input = tool_input.split(",")
        log(multi_input)

        # Cast max_results to int
        try:
            max_results = int(multi_input[1])
        except ValueError:
            # If the model leave a quote remove it
            max_results = int(multi_input[1].strip("'"))

        # Query for PubMed
        query = f"{multi_input[0]}[Title]"

        return query, max_results

    def parse_results(self, results):
        cleaned = []

        # Loop all results
        for result in results:
            # TODO check that results is not empty
            string = ""

            # Make Dict
            r = result.toDict()

            # Drop useless keys
            r.pop("xml")
            r.pop("pubmed_id")

            # Loop keys
            for key in r.keys():

                # Make a string
                string += f"**{key}**: {r[key]}\n"

        cleaned.append(string)

        return cleaned

    def __query(self, query: str, max_results: int = 1):

        # Query PubMed
        results = self.pymed.query(query=query, max_results=max_results)

        # Store docs in Working Memory for further operations.
        # e.g. filter docs
        self.cat.working_memory["pubmed_results"] = self.parse_results(results)

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
         - PUBMED: Look for "Public Healthcare" and give me 3 results. Input is 'Public Healthcare,3'
         - PUBMED: Look for "Antibiotic resistance". Input is 'Public Healthcare,1'
    """

    # Schrodinger Cat
    schrodinger_cat = SchrodingerCat(cat)

    # Search on PubMed
    schrodinger_cat.make_search(query)

    # TODO: change this output
    out = f"Alright. I'm looking for {schrodinger_cat.parse_query(query)[1]} results about" \
          f" {schrodinger_cat.parse_query(query)[0].strip('[Title]')} on PubMed. This may take some time. " \
          f"Hang on please, I'll tell you when I'm done"

    return out


@tool()
def empty_working_memory(tool_input, cat):
    """
    Useful to empty and forget all the documents in the Working Memory. Input is always None.
    """
    if "pubmed_results" in cat.working_memory:
        cat.working_memory.pop("pubmed_results")

    # TODO: this has to be tested
    # the idea is having the Cat answer without directly returning a hard coded output string
    return cat.llm("Can you forget everything I asked you to keep in mind?")


@tool(return_direct=True)
def summary_working_memory(tool_input, cat):
    """
    Useful to ask for a detailed summary of what's in the Cat's Working Memory. Input is always None.
    Example:
        - What's in your memory? -> use summary_working_memory tool
        - Tell me the papers you have currently in memory
    """
    # Memories in Working Memory
    if "pubmed_results" in cat.working_memory.keys():
        memories = cat.working_memory["pubmed_results"]

        n_memories = len(memories)

    else:
        memories = []
        n_memories = 0

    if n_memories == 0:
        return cat.llm("Say that you memory is empty")

    prefix = f"Currently I have {n_memories} papers temporarily loaded in memory.\n"
    papers = ""
    for m in memories:
        papers += f"{m}\n"
    log(papers)

    return prefix + papers


# @tool
# def query_memory(tool_input, cat):
#     """
#     Useful to query the currently stored paper about a specific question. Input is a string with a question.
#     """
#     tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b")
#     model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b")
#
#     if "pubmed_results" in cat.working_memory.keys():
#         memories = cat.working_memory["pubmed_results"]
#
#     input_text = tool_input + memories + " [START_REF]"
#     input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#
#     outputs = model.generate(input_ids)
#     log(tokenizer.decode(outputs[0]))
#
#     return tokenizer.decode(outputs[0])


# @tool(return_direct=True)
# def explain_paper(tool_input, cat):
#     """
#     Useful to have a paper explained in a systematic way.
#     The Cat answers 10 questions from
#     'Carey MA, Steiner KL, Petri WA Jr. Ten simple rules for reading a scientific paper'
#     Input to this tool is the title of a paper.
#     """
#
#     # Schrodinger Cat
#     scat = SchrodingerCat(cat)
#
#     # Set paper to none
#     paper = None
#
#     # Look for the Title of the paper in Working Memory
#     for m in scat.working_memory.memories:
#         if m.metadata['Title of this paper'] == tool_input:
#             paper = m
#
#     # If not in Worming Memory, search in Long Term Memory
#     if paper is None:
#         paper = cat.memory.vectors.declarative.similarity_search(
#             query=tool_input,
#             k=1,
#             filter={"Title of this paper": tool_input}
#         )
#
#     # If not in Long Term Memory return telling it doesn't exist
#     if paper is None:
#         return "I don't have the paper you queried for"
#
#     # Text splitter (big chunks good idea?)
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=500,
#                                                    separators=["\\n\\n", "\n\n", ".\\n", ".\n", "\\n", "\n", " ", ""])
#
#     # Split the document
#     docs = text_splitter.split_documents([paper])
#
#     # Log info - to be deleted
#     log(len(docs))
#
#     # docs = list(filter(lambda d: len(d.page_content) > 10, docs))
#
#     # Questions from 'Carey MA, Steiner KL, Petri WA Jr. Ten simple rules for reading a scientific paper'
#     questions = [
#         'What do the authors want to know? What is the motivation?',
#         'What did the authors do? What is the approach/methods?',
#         'Why was the approach done that way? Which is the context within the field of research?',
#         'What do the results show?',
#         'How did the authors interpret the results? Which is their discussion?',
#         'What should be done next? The authors may provide some suggestions in the discussion'
#     ]
#
#     # Summarize with refine because most lossless chain
#     chain = load_summarize_chain(cat.llm, chain_type="refine")
#
#     # Summarize
#     summaries = [chain.run([d]) for d in docs]
#
#     # Log info - to be deleted
#     log(summaries)
#
#     # Make document from summaries - metadata?
#     s = [Document(page_content=ss) for ss in summaries]
#
#     # llm = HuggingFaceHub(repo_id='microsoft/BioGPT-Large-PubMedQA',
#     #                      huggingfacehub_api_token='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')
#
#     # Question & Answer chain
#     # TODO two refine takes ages, test stuff in q&a
#     qa_chain = load_qa_chain(cat.llm, chain_type="stuff")  # cambia chain type
#
#     # Gather answers
#     answers = [qa_chain({"input_documents": s, "question": q}) for q in questions]
#     log(answers)
#     if 'authors' in paper.metadata.keys():
#         authors = [f"{paper.metadata['authors'][i]['lastname']} {paper.metadata['authors'][i]['initials']}."
#                    for i in range(len(paper.metadata['authors']))]
#         full_authors = ",".join(authors)
#         prefix = f"## {paper.metadata['Title of this paper']}\n### {full_authors}\n\n"
#     else:
#         prefix = f"## {paper.metadata['Title of this paper']}\n\n"
#
#     if 'doi' in paper.metadata.keys():
#         suffix = f"URL:[{paper.metadata['url']}]({{paper.metadata['url']}})"
#     else:
#         suffix = ""
#
#     body = ""
#     for q, a in zip(questions, answers):
#         body += f"**{q}**\n{a['output_text']}\n\n"
#
#     return prefix + body + suffix


