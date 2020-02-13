import defaults

def initiate() -> None:
    if defaults.LOG_SAVE:
        if not defaults.LOG_SAVE_DESTINATION.exists():
            defaults.LOG_SAVE_DESTINATION.parent.mkdir(parents=True, exist_ok=True)
        # Making sure the logfile is empty
        open(defaults.LOG_SAVE_DESTINATION, "w").close()

def tell(line: str) -> None:
    if defaults.VERBOSE:
        print(line)
    if defaults.LOG_SAVE:
        with open(defaults.LOG_SAVE_DESTINATION, "a+") as LOGFILE:
            LOGFILE.write(line + '\r\n')