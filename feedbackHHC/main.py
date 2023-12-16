from read_data import ReadData


def main():

    list_nr_cols_to_be_eliminated = [2, 3, 4, 6, 7, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 50, 51, 58, 59, 66, 67, 69]
    new_list = [i-1 for i in list_nr_cols_to_be_eliminated]

    text_file_name = 'dataset/HH_Provider_Oct2023.csv'
    rd = ReadData(text_file_name)
    rd.elim_cols(new_list)
    # print(rd.get_dataframe())
    rd.elim_rows()

    rd.elim_outliers_and_replace_with_mean()
    #print(rd.get_dataframe().iloc[:, 4])

    #rd.elim_rows_with_()
    #rd.print_cols_names()


if __name__ == '__main__':
    main()
