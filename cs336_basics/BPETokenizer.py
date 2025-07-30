import regex
import collections
import multiprocessing
import time
import os

import os
from typing import BinaryIO
from typing import Iterable, Iterator

import json


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


gpt2_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        pat_str: str | None = None,
        special_tokens: list[str] | None = None,
    ):
        #self.vocab_size = vocab_size
        self.pat_str = pat_str if pat_str else gpt2_pattern
        self.special_tokens = sorted(special_tokens, key=len, reverse= True) if special_tokens else []
        
        self.vocab = vocab if vocab else {}
        self.merges = merges if merges else []
        self.mergeable_ranks = {v: k for k, v in self.vocab.items()} if self.vocab else {}
    
    @classmethod
    def from_files(
        cls, 
        vocab_filepath: str,
        merges_filepath: str, 
        special_tokens: list[str] | None = None
    ):
        """
        Load a BPE tokenizer from vocab and merges files.
        """

        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
            # Convert vocab keys to int if they are not already
            vocab = {int(v): bytes(k, "utf-8") for k, v in vocab.items()}
        
        
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [tuple(bytes(x, "utf-8") for x in line.split()) for line in f.readlines()[1:]]
        
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def fast_train(
        self,
        words: list[list[bytes]],
        vocab_size: int,
        pat_str: str,
    ):
        ranks = {}
        for i in range(2**8):
            ranks[bytes([i])] = i
        merges = []
        current_word_tokens: list[list[bytes]] = [list(piece) for piece in words]
        pair_stats = collections.defaultdict(lambda: {"count": 0, 'location': set()})
        for i, piece in enumerate(current_word_tokens):
            for j in range(len(piece) - 1):
                pair = (piece[j], piece[j + 1])
                pair_stats[pair]["count"] += 1
                pair_stats[pair]["location"].add((i, j))
        
        #print(f"Initial pair stats: {pair_stats}")
        while len(ranks) < vocab_size:
            most_common_pair = max(pair_stats, key=lambda pair: (pair_stats[pair]["count"], pair))
            #print(f"Most common pair: {most_common_pair} with count {pair_stats[most_common_pair]['count']}")
            p1, p2 = most_common_pair
            token_bytes = p1 + p2
            token = len(ranks)
            ranks[token_bytes] = token
            merges.append(most_common_pair)

            loc_list_to_update = pair_stats[most_common_pair]["location"]
            pair_stats.pop(most_common_pair, None)
            for i, j in loc_list_to_update:
                piece = current_word_tokens[i]
                new_piece = []
                k = 0
                while k < len(piece) - 1:
                    if (piece[k], piece[k + 1]) == most_common_pair:
                        new_piece.append(token_bytes)
                        if k > 0:
                            old_left_pair = (piece[k-1], piece[k])    #update old left pair
                            if old_left_pair in pair_stats:
                                pair_stats[old_left_pair]["location"].discard((i, k - 1))
                                pair_stats[old_left_pair]["count"] -= 1
                        if k < len(piece) - 2:
                            old_right_pair = (piece[k+1], piece[k + 2]) #update old right pair
                            if old_right_pair in pair_stats:
                                pair_stats[old_right_pair]["location"].discard((i, k + 1))
                                pair_stats[old_right_pair]["count"] -= 1
                        
                        if k > 0:
                            new_left_pair = (piece[k - 1], token_bytes)  #add new left pair
                            if new_left_pair in pair_stats:
                                pair_stats[new_left_pair]["location"].add((i, k - 1))
                                pair_stats[new_left_pair]["count"] += 1
                            else:
                                pair_stats[new_left_pair] = {"count": 1, "location": {(i, k - 1)}}
                        if k < len(piece) - 2:
                            new_right_pair = (token_bytes, piece[k + 2])   #add new right pair
                            if new_right_pair in pair_stats:
                                pair_stats[new_right_pair]["location"].add((i, k + 1))
                                pair_stats[new_right_pair]["count"] += 1
                            else:
                                pair_stats[new_right_pair] = {"count": 1, "location": {(i, k + 1)}}
                            
                        k += 2

                    else:
                        new_piece.append(piece[k])
                        k += 1
                if k == len(piece) - 1:
                    new_piece.append(piece[k])
                current_word_tokens[i] = new_piece
        vocab = {v: k for k, v in ranks.items()}
        #print(f"Final ranks: {ranks}")
        #print(f"Final words: {current_word_tokens}")
        return vocab, merges    

    def train(self, words, vocab_size:int, pat_str:str):
        ranks = {}
        for i in range(2**8):
            ranks[bytes([i])] = i
        merges = []
        while len(ranks) < vocab_size :
            stats = collections.Counter()
            for piece in words:
                for pair in zip(piece[:-1], piece[1:]):
                    stats[pair] += 1
            most_common_pair = max(stats, key=lambda pair: (stats[pair], pair))
            #print(f"Most common pair: {most_common_pair} with count {stats[most_common_pair]}")
            token_bytes = most_common_pair[0] + most_common_pair[1]
            token = len(ranks)
            ranks[token_bytes] = token
            merges.append(most_common_pair)

            new_words = []
            for word in words:
                new_word = []
                i = 0
                while i < len(word) - 1:
                    pair = (word[i], word[i + 1])
                    if pair == most_common_pair:
                        new_word.append(token_bytes)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                if i == len(word) - 1:
                    new_word.append(word[i])
                new_words.append(new_word)
            words = new_words
            #print(f"Final ranks: {ranks}")
            #print(f"Final words: {words}") 
        vocab = {v: k for k, v in ranks.items()}
        return vocab, merges

    def process_chunk(self, chunk: str):
        if self.special_tokens:
            special_pattern = "(" + "|".join(regex.escape(k) for k in self.special_tokens) + ")"
        #print(special_pattern)
        chunk = regex.split(special_pattern, chunk)
        #print(f"Processed chunk: {chunk}")
        words =[]
        for i, part in enumerate(chunk):
            if i % 2 == 0:
                if part:
                    for match in regex.finditer(self.pat_str, part):
                        words.append([bytes([b]) for b in match.group(0).encode("utf-8")])
        #print(f"Words in chunk: {words}")
        return words

    def _encode_ordinary_text(self, text) -> list[int]:
        ids = []
        tokens = regex.findall(self.pat_str, text)
        for match in tokens:
            token_bytes = [bytes([b]) for b in match.encode("utf-8")]
            # print(f"Token bytes: {token_bytes}")

            while True:
                min_idx = None
                min_rank = None

                for i, pair in enumerate(zip(token_bytes[:-1], token_bytes[1:])):
                    rank = self.mergeable_ranks.get(pair[0] + pair[1], None)
                    if rank is not None and (min_rank is None or rank < min_rank):
                        min_idx = i
                        min_rank = rank
                if min_idx is None:
                    break

                token_bytes = (
                    token_bytes[:min_idx]
                    + [token_bytes[min_idx] + token_bytes[min_idx + 1]]
                    + token_bytes[min_idx + 2:]
                )
                # print(f"Token bytes after merge: {token_bytes}")

            # print(f"Final token bytes: {token_bytes}")
            ids.extend(self.mergeable_ranks[b] for b in token_bytes if b in self.mergeable_ranks)
        return ids


    def encode(self, text: str) -> list[int]:
        ids = []
        special_tokens = [token.encode("utf-8") for token in self.special_tokens]
        #print(f"Special tokens: {special_tokens}")
        special = {v: k for k, v in self.vocab.items() if v in special_tokens}
        #print(f"Special tokens: {special}")
        self.mergeable_ranks = {v: k for k, v in self.vocab.items()}
        special_pattern = "(" + "|".join(regex.escape(k) for k in self.special_tokens) + ")"
        if len(self.special_tokens) == 0:
            return self._encode_ordinary_text(text)
        spcial_chunks = regex.split(special_pattern, text)
        #print(f"Special chunks: {spcial_chunks}")

        for part in spcial_chunks:
            #print(f"Processing part: {part}")
            if part in self.special_tokens:
                ids.append(special[part.encode("utf-8")])
            else:
                part_ids = self._encode_ordinary_text(part)
                ids.extend(part_ids)         
        return ids
                
    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        """
        all_bytes = b"".join(self.vocab[i] for i in ids if i in self.vocab)
        return all_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:

        if len(self.special_tokens) == 0:
            special_pattern = None
        else:
            special_pattern = "(" + "|".join(regex.escape(k) for k in self.special_tokens) + ")"
        
        for text in iterable:
            if not special_pattern:
                ids_for_line = self._encode_ordinary_text(text)
                for token_id in ids_for_line:
                    yield token_id
            else:
                test_chunks = regex.split(special_pattern, text)
                for part in test_chunks:
                    if part in self.special_tokens:
                        yield self.mergeable_ranks[part.encode("utf-8")]
                    else:
                        ids_for_line = self._encode_ordinary_text(part)
                        for token_id in ids_for_line:
                            yield token_id

    def pretokenize(
        self,
        input_path:str,
        num_processes:int,
        split_special_token: str
    ):
        all_pre_tokens = []
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_processes, split_special_token.encode("utf-8"))

            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            tasks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                tasks.append(chunk)
            
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(self.process_chunk, tasks)
            for result in results:
                all_pre_tokens.extend(result)
        return all_pre_tokens
        #print(f"All pre-tokens: {all_pre_tokens}")
                
    def bpe_train(
        self,
        input_path: str,
        vocab_size: int,
        special_tokens: list[str] | None = None
    ):
        vocab_size = vocab_size - len(special_tokens) if special_tokens else vocab_size
        pre_tokens = self.pretokenize(
            input_path=input_path,
            num_processes= 4,
            split_special_token="<|endoftext|>",
        )
        vocab, merges = self.fast_train(
            words=pre_tokens,
            vocab_size=vocab_size,
            pat_str=self.pat_str,
        )
        for i in range(vocab_size, vocab_size + len(special_tokens)):
            token = bytes(special_tokens[i - vocab_size], "utf-8")
            vocab[i] = token
        #print(merges)
        self.vocab = vocab
        self.merges = merges
        return vocab, merges


            

if __name__ == '__main__':
    """
    multiprocessing.freeze_support()  # Ensure multiprocessing works on Windows  
         
    bpe = BPETokenizer(special_tokens=["<|endoftext|>"])
    bpe.bpe_train(
        input_path="/Users/yfcheng/Desktop/cs336/assignment1-basics/tests/fixtures/corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    """
    MERGES_FILE = "/Users/yfcheng/Desktop/project/assignment1-basics/tests/fixtures/gpt2_merges.txt"
    VOCAB_FILE = "/Users/yfcheng/Desktop/project/assignment1-basics/tests/fixtures/gpt2_vocab.json"
    bpe = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_FILE,
        merges_path=MERGES_FILE,
        special_tokens=["<|endoftext|>"],
    )
    text = "Hello, world! 🙃This is a test. <|endoftext|> Let's see how it works."
    encoded = bpe.encode(text)
    decoded = bpe.decode(encoded)
    