lines = open(r'C:\Users\vaibh\Desktop\aeronet_ai\services\evaluation_service.py', encoding='utf-8').read().split('\n')
print(repr(lines[258])); print(repr(lines[273]))
del lines[258:274]
open(r'C:\Users\vaibh\Desktop\aeronet_ai\services\evaluation_service.py','w',encoding='utf-8').write('\n'.join(lines))
print('new total lines:', len(lines))

