import ujson as json

def fancyprint(in_str):
    print()
    print("#"*20)
    print("# " + in_str)
    print("#"*20)
    print()

def save(filename, obj, message=None):
    """
    just saves the file, nothing fancy
    author: @wzhouad
    """
    if message is not None:
        fancyprint("Saving {}!".format(message))
        with open(filename, "w") as fh:
            json.dump(obj, fh)