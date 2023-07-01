import os


class FileOperations:
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def get_directory_names(self):
        directory_names = []
        for root, dirs, files in os.walk(self.directory_path):
            for directory in dirs:
                directory_names.append(directory)
        return directory_names

    def read_file_lines(self, file_path):
        lines = []
        with open(file_path, "r") as file:
            lines = file.readlines()
        return lines

    def create_description_dict(self, file_path):
        directory_names = self.get_directory_names()
        lines = self.read_file_lines(file_path)

        descriptions = {}
        line_index = 0
        names_len = len(directory_names)

        for i in range(0, names_len, 2):
            name = directory_names[i]
            if i + 1 < names_len:
                next_name = directory_names[i + 1]
                descriptions[name] = lines[line_index].strip()
                descriptions[next_name] = lines[line_index].strip()
                line_index += 1

        return descriptions
