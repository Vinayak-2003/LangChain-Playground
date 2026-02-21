from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
There's something oddly satisfying about rearranging your room at midnight. What starts as a simple idea—“I'll just move the desk a little”—quickly turns into a full transformation. Suddenly, you're rediscovering old notebooks, untangling mysterious cables, and questioning why you kept that one random receipt from three years ago. By the time you're done, the room feels brand new, even though nothing actually changed except the position of the furniture.

Street food adventures are another underrated thrill. One evening you decide to “just try one snack,” and before you know it, you're balancing a plate of spicy noodles in one hand and a chilled dessert in the other. The best part isn't just the food—it's the chaos, the laughter, and the shared excitement of discovering something delicious and completely unexpected.

And then there's the universal joy of finding money in an old pocket. It could be a tiny amount, but it feels like you just unlocked a secret reward from your past self. For a brief moment, you become richer, luckier, and slightly more convinced that the universe occasionally sends small, delightful surprises your way.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

print("len of chunks:", len(chunks))
print(chunks)
