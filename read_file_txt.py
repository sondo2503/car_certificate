# def rel(s):
#     s = s.replace('1)', '')
#     s = s.replace('2)', '')
#     s = s.replace('3)', '')
#     s = s.replace('4)', '')
#     s = s.replace('1.)', '')
#     s = s.replace('2.)', '')
#     s = s.replace('3.)', '')
#     s = s.replace('4.)', '')
#     return s
#
# with open(r'T:\Competi\ocr_ai_pill\txt_file_name.txt',encoding='utf-8') as file:
#     for line in file:
#         print(rel(line).rstrip())
#         with open(r'T:\Competi\ocr_ai_pill\id_drug.txt','a', encoding='utf-8') as file2:
#             file2.writelines(rel(line).rstrip())
#             file2.writelines('\n')
#             file2.close()
with open('id_drug.txt', encoding='utf-8') as fl:
    content = fl.read().split('\n')

content = set([line for line in content if line != ''])

content = '\n'.join(content)

with open('id_drug.txt', 'w', encoding='utf-8') as fl:
    fl.writelines(content)