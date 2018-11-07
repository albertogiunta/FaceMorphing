def print_status(i):
    if i == 0 or (i + 1) % 10 == 0:
        from time import gmtime, strftime
        if i != 0:
            i += 1
        print("\t\t\t\t{} - {}".format(i, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
