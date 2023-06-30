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
        for name in directory_names:
            if lines:
                descriptions[name] = lines.pop(0).strip()

        return descriptions