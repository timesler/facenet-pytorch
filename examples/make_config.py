import configparser


def create_config(path):
    """
    Create a config file
    """
    config = configparser.ConfigParser()
    config.add_section("Parameters")
    config.set("Parameters", "epochs", "10")
    config.set("Parameters", "bath_size", "64")
    config.set("Parameters", "AMP", "False")
    config.set("Parameters", "training_name", "1")
    config.set("Parameters", "data_dir", "data")

    with open(path, "w") as config_file:
        config.write(config_file)


if __name__ == '__main__':
    create_config('cfg.txt')
    print("Config created")
