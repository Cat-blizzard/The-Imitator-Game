import numpy as np
import pytest
from PIL import Image
from resimpair_assets.source_mesh import resources,load


def test_gso_texture_directory_is_resolved_and_missing_texture_is_not_gray_success(tmp_path):
    meshes=tmp_path/'meshes';meshes.mkdir()
    textures=tmp_path/'materials/textures';textures.mkdir(parents=True)
    obj=meshes/'model.obj'
    obj.write_text('mtllib model.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nusemtl example\nf 1/1 2/2 3/3\n')
    (meshes/'model.mtl').write_text('newmtl example\nKd 1 1 1\nmap_Kd texture.png\n')
    image=textures/'texture.png';Image.new('RGB',(8,8),(255,80,20)).save(image)
    row={'source':'google_scanned_objects_original_fuel','primary_path':str(obj)}
    manifest=resources(row);assert manifest['declared_texture_count']==1 and not manifest['missing']
    scene,evidence=load(row,manifest)
    assert evidence['textured_geometry_count']==1
    assert any(np.asarray(g.visual.material.image)[0,0,0]==255 for g in scene.geometry.values())
    image.unlink()
    assert resources(row)['missing']==['texture.png']
    with pytest.raises(ValueError,match='Missing mesh resources'):load(row)


def test_packaged_products_are_not_single_object_semantic_certificates():
    from resimpair_assets.catalog import classify
    from resimpair_grasp.common import ROOT,read
    policy=read(ROOT/'configs/asset_roles_v4.json')
    for name in ['Diet Pepsi Soda Cola12 Pack 12 oz Cans','Nestle Water Dragon Fruit 8 count 12 fl oz can','LEGO City Advent Calendar']:
        result,_=classify({'kind':'model','name':name,'categories':'Consumer Goods'},policy)
        assert result['semantic_ambiguities'] and not result['manipulation_candidate']
