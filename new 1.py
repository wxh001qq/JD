#层级六种行为次数
def get_action_level_feat1(start_date,end_date):
    dump_path = './cache/level_feat1_%s_%s.csv' % (start_date, end_date)
    if os.path.exists(dump_path):
        action = pd.read_csv(dump_path)
    else:
        actions=get_actions(start_date,end_date)[['sku_id','time','type']]
        actions['time']=actions['time'].map(lambda x:datetime.strptime(end_date,'%Y-%m-%d')-datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
        actions['day']=actions['time'].dt.days
        del actions['time']
        df = pd.get_dummies(actions['type'], prefix='%s-%s-action' % (start_date, end_date))
        del actions['type']
        actions = pd.concat([actions, df], axis=1)
        action=None
        for i in (3,7,14,28):
            df = actions
            df['day_'+str(i)]=df['day'].map(lambda x: x/i )
            df=df.drop(['day'],axis=1)
            grouped=df.groupby(['sku_id','day_'+str(i)]).sum()
            grouped=grouped.unstack()
            grouped.columns=[range(grouped.shape[1])]
            grouped=grouped.reset_index()
            del actions['day_'+str(i)]
            if action is None:
                action=grouped
            else:
                action=pd.merge(action,grouped,on='sku_id',how='outer')
            action.to_csv(dump_path,index=False)
    return action

	
print 