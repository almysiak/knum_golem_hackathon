def insert_predictions_to_json(df, ref_json):
    final = pd.DataFrame()
    final = final.assign(id = df['id'])
    final = final.assign(category_id = df['category_id'])
    anno = ref_json['annotations']
    for index in range(len(anno)):
        anno_id = anno[index]['id']
        cat_id = final[final.id==anno_id]['category_id'].values[0]
        anno[index]['category_id'] = cat_id
    ref_json['annotations'] = anno
    return ref_json