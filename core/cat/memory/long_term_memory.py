import time

from cat.memory.vector_memory import VectorMemory
from cat.utils import log

# This class represents the Cat long term memory (content the cat saves on disk).
class LongTermMemory:
    def __init__(self, vector_memory_config={}):
        # Vector based memory (will store embeddings and their metadata)
        self.vectors = VectorMemory(**vector_memory_config)

        # What type of memory is coming next?
        # Surprise surprise, my dear!


class WorkingMemory:
    def __init__(self):
        self.memories = []

    def keep_in_mind(self, information):
        if len(information) == 1:
            self.memories.append(information[0])
        else:
            self.memories.extend(information)

    def forget(self):
        self.memories = []

    def remember(self, ccat, source, **kwargs):
        for m, mem in enumerate(self.memories):
            _ = ccat.memory.vectors.declarative.add_texts(
                [mem.page_content],
                [
                    {
                        "source": source,
                        "when": time.time(),
                        "text": mem.page_content,
                        **mem.metadata,
                        **kwargs
                    }
                ],
            )
            log(f"Inserted into memory ({m + 1}/{len(mem)}):    {mem.page_content}")
            time.sleep(0.1)
