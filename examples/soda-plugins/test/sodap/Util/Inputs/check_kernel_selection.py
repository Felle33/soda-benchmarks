import json,sys
old,new=[json.load(open(p)) for p in sys.argv[1:]]
for report in [old,new]:
 valid=[(c['cost']['dma_cycles'],c['cost']['compute_iterations'],c['cost']['plm_bytes'],i)
        for i,c in enumerate(report['candidates']) if c['status']=='VALID']
 assert report['selected']==min(valid)[-1]
 assert all(c['cost']['plm_bytes']<=report['plm_budget'] for c in report['candidates'] if c['status']=='VALID')
 assert isinstance(report['candidates'][report['selected']]['tiles'],list)
assert new['schedule']=='nonoverlap' and old['schedule']=='overlap'
assert new['old_overlap_selected']==old['selected']
for a,b in zip(old['candidates'],new['old_overlap_candidates']):
 assert a['status']==b['status']
 if a['status']=='VALID': assert a['cost']==b['cost'] and a['tiles']==b['tiles']
assert any(b['cost']['plm_bytes']<a['cost']['plm_bytes'] for a,b in zip(old['candidates'],new['candidates']) if a['status']==b['status']=='VALID')
