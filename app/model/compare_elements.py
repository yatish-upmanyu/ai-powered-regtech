def compare_elements(elements_dict1, elements_dict2):
    all_keys = set(elements_dict1.keys()).union(set(elements_dict2.keys()))
    report = []

    for key in all_keys:
        type1, annotation1 = elements_dict1.get(key, (None, None))
        type2, annotation2 = elements_dict2.get(key, (None, None))

        if key not in elements_dict2:
            report.append(('Removed', key, type1, None, annotation1))
        elif key not in elements_dict1:
            report.append(('Added', key, None, type2, annotation2))
        elif type1 != type2:
            report.append(('Modified', key, type1, type2, annotation2))
        elif annotation1 != annotation2 and annotation2 == 'yellow field':
            report.append(('Annotation Changed', key, type1, type2, annotation2))

    return report
