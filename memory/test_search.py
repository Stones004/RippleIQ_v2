# Run it like this 
# python -m memory.test_search


from precedent_search import PrecedentSearch


search = PrecedentSearch()

results = search.search(
    "AI changing labor markets"
)

for result in results:

    print("\n=================")

    print(
        f"TITLE: {result['title']}"
    )

    print(
        f"SCORE: {result['score']:.4f}"
    )

    print(
        f"NODE: {result['canonical_node']}"
    )

    print("=================\n")