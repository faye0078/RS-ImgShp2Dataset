def get_id_names(id, file_list):
    id_name_list = []
    for file in file_list:
        if file.split("_")[1] == id:
            id_name_list.append(file)
    return id_name_list