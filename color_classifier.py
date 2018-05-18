import color_import

def run():
    print("Importing color data...")
    color_data = color_import.import_data()
    print("Normalizing color data...")
    color_data = color_import.normalize_data(color_data)

    print(color_data.head(10))

run()