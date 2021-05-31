import os

if __name__ =='__main__':
    data_base = 'dataset/tagging/tagging_dataset_test_5k/'
    modal_type = ('video_npy/Youtube8M/tagging/', 'audio_npy/Vggish/tagging/', 'image_jpg/tagging/', 'text_txt/tagging/')
    modal_suffix = ('.npy', '.npy', '.jpg', '.txt')
    data_path = [data_base + x for x in modal_type]
    lines = []
    for data in os.listdir(data_path[0]):
        filename = data.split(modal_suffix[0])[0]
        for i in range(4):
            lines.append('../' + data_path[i] + filename + modal_suffix[i] + '\n')
        lines.append('\n')
        lines.append('\n')
    # print(lines)
    with open('dataset/tagging/GroundTruth/datafile/test.txt', 'w') as f:
        f.writelines(lines)


