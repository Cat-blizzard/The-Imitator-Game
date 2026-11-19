import numpy as np

from resimpair_pickplace.visual_proxy_metrics_v1 import geometry,rgb_metrics


def fixture_image(color=128):
    rgb=np.full((96,96,3),color,np.uint8);mask=np.zeros((96,96),bool);mask[40:56,40:56]=True
    table=~mask;seg=np.where(mask,2,1)
    return rgb,mask,table,seg


def test_same_color_has_no_local_color_contrast_but_visible_pixels():
    rgb,mask,table,seg=fixture_image()
    result=rgb_metrics(rgb,mask,table,table,seg)
    assert result['pixel_geometry']['visible_pixels']==256
    assert result['local_context_contrast']['median_color_deltaE76']==0
    assert result['local_context_contrast']['luminance_CNR']==0


def test_actual_RGB_changes_contrast_without_changing_segmentation_size():
    rgb,mask,table,seg=fixture_image(20);rgb[mask]=240
    result=rgb_metrics(rgb,mask,table,table,seg)
    assert result['pixel_geometry']['visible_pixels']==256
    assert result['local_context_contrast']['median_luminance_difference_L']>80
    assert result['local_context_contrast']['median_color_deltaE76']>80


def test_table_texture_uses_RGB_not_segmentation_edges():
    rgb,mask,table,seg=fixture_image(128);rgb[mask]=0
    flat=rgb_metrics(rgb,mask,table,table,seg)['table_texture']
    assert flat['highpass']['1.0']['RMS_L']<1e-3
    checker=np.indices(mask.shape).sum(axis=0)%2
    rgb[table]=np.repeat((checker[table]*220+20)[:,None],3,axis=1)
    busy=rgb_metrics(rgb,mask,table,table,seg)['table_texture']
    assert busy['highpass']['1.0']['RMS_L']>20


def test_diagonal_thin_object_is_not_mistaken_for_thick_bbox():
    mask=np.eye(40,dtype=bool);g=geometry(mask)
    assert g['bbox_short_edge_px']==40
    assert g['pca_minor_width_px']<1.001


def test_absent_pixels_remain_unknown_not_good_contrast():
    rgb,mask,table,seg=fixture_image();mask[:]=False
    result=rgb_metrics(rgb,mask,table,table,seg)
    assert result['local_context_contrast']['status']=='unknown'
    assert result['pixel_geometry']['visible_pixels']==0
