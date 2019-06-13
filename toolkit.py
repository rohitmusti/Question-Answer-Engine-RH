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

def quick_clean(raw_str):
    """
    args:
        - context: a string to be quickly cleaned

    return
        - the original string w/ all quotes replaced as double quotes
    """
    return raw_str.replace("''", '" ').replace("``", '" ')