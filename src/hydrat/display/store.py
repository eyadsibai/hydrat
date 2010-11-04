from __future__ import with_statement
import os

from hydrat.display.html import TableSort
from hydrat.common.log import getHydraLogger
from hydrat.display import show_metadata
import hydrat.display.sparklines as sparklines

from scipy.sparse import issparse


def store_report(store, outfile, ds_report_dir = None):
  with TableSort(outfile) as index_renderer:
    # Datasets
    index_renderer.section('Datasets')
    rows = []
    for ds_name in sorted(store.list_Datasets()):
      ds_tag = store.resolve_Dataset(ds_name)
      metadata = store.get_Metadata(ds_tag)
      row = dict(metadata)
      class_spaces = store.list_ClassSpaces(ds_name)
      feature_spaces = store.list_FeatureSpaces(ds_name)
      row['uuid'] = str(row['uuid'])
      row['cl_count'] = len(class_spaces)
      row['ft_count'] = len(feature_spaces)
      if ds_report_dir is not None:
        ds_report_relpath = os.path.join(ds_report_dir, row['uuid'] + '.html')
        ds_report_path = os.path.join(os.path.dirname(outfile.name), ds_report_relpath) 
        dataset_report(store, ds_name, open(ds_report_path, 'w'))
        row['link'] = '<a href="%s">link</a>' % ds_report_relpath
      rows.append(row)

    relevant = [ ("Name", 'name')
              , ("# Instances", 'num_instances')
              , ("# Class Spaces", 'cl_count' )
              , ("# Feature Spaces", 'ft_count' )
              , ("Date", 'date')
              , ("UUID", 'uuid')
              ]
    if ds_report_dir is not None:
      relevant += [ ("Details", 'link') ]

    headings, cols = zip(*relevant)
    index_renderer.dict_table(rows, cols, col_headings = headings)

    # Class Spaces
    index_renderer.section('Class Spaces')
    rows = []
    for name in sorted(store.list_ClassSpaces()):
      row = {}
      tag = store.resolve_Space({'type':'class', 'name':name})
      space = store.get_Space(tag)
      metadata = store.get_Metadata(tag)

      row['name'] = name
      row['tag']  = tag
      row['length'] = metadata['size'] 
      row['date']  = metadata['date']
      rows.append(row)
    relevant = [ ("Name", 'name')
              , ("# Classes", 'length')
              , ("Date", 'date')
              , ("UUID", 'tag')
              ]

    headings, cols = zip(*relevant)
    index_renderer.dict_table(rows, cols, col_headings = headings)

    # Feature Spaces
    index_renderer.section('Feature Spaces')
    rows = []
    for name in sorted(store.list_FeatureSpaces()):
      row = {}
      tag = store.resolve_Space({'type':'feature', 'name':name})
      space = store.get_Space(tag)
      metadata = store.get_Metadata(tag)

      row['name'] = name
      row['tag']  = tag
      row['length'] = metadata['size']
      row['date']  = metadata['date']
      rows.append(row)
    relevant = [ ("Name", 'name')
              , ("# Features", 'length')
              , ("Date", 'date')
              , ("UUID", 'tag')
              ]

    headings, cols = zip(*relevant)
    index_renderer.dict_table(rows, cols, col_headings = headings)

    # TaskSets
    # TODO: Come up with a nice report for this!!!

def dataset_report(store, ds_name, outfile):
  ds_tag = store.resolve_Dataset(ds_name)
  ds_metadata = store.get_Metadata(ds_tag)
  with TableSort(outfile) as renderer:
    renderer.section("Dataset: %s" % ds_name)
    show_metadata(renderer, ds_metadata)

    # Class Spaces
    renderer.section('Class Spaces')
    rows = []
    for name in sorted(store.list_ClassSpaces(ds_name)):
      row = {}
      tag = store.resolve_Space({'type':'class', 'name':name})
      metadata = store.get_Metadata(tag)
      class_map = store.get_ClassMap(ds_tag, tag)
      class_distrib = class_map.raw.sum(axis=0)

      row['name'] = name
      row['class_dist'] = '<img src="%s"/>' % sparklines.barchart(class_distrib, width=1, gap=1)
      row['num_classes'] = metadata['size']
      rows.append(row)
    relevant = [ ("Name", 'name')
               , ("# Classes", 'num_classes')
               , ("Class Distribution", 'class_dist')
               ]

    headings, cols = zip(*relevant)
    renderer.dict_table(rows, cols, col_headings = headings)

    # Feature Spaces
    renderer.section('Feature Spaces')
    rows = []
    for name in sorted(store.list_FeatureSpaces(ds_name)):
      row = {}
      tag = store.resolve_Space({'type':'feature', 'name':name})
      metadata = store.get_Metadata(tag)
      instance_sizes = store.get_SizeData(ds_tag, tag)
      feature_map = store.get_FeatureMap(ds_tag, tag)
      feature_distrib = feature_map.raw.sum(axis=0)
      row['size_dist'] = '<img src="%s"/>' % sparklines.histogram(instance_sizes)
      row['feat_dist'] = '<img src="%s"/>' % sparklines.histogram(feature_distrib)
      row['density'] = feature_map.raw.size / float(feature_map.raw.shape[0] * feature_map.raw.shape[1])
        

      row['name'] = name
      row['num_features'] = metadata['size']
      row['tokens'] = instance_sizes.sum()
      rows.append(row)
    relevant = [ ("Name", 'name')
               , ("# Features", 'num_features')
               , ("Density", 'density')
               , ("Total Tokens", 'tokens')
               , ("Instance Size", 'size_dist')
               , ("Overall Feature Count", 'feat_dist')
               ]

    headings, cols = zip(*relevant)
    renderer.dict_table(rows, cols, col_headings = headings)

def write_report(store, report_dir='html'):
  ds_report_dir = 'datasets'
  # Create report dir if needed
  if not os.path.exists(report_dir):
    os.mkdir(report_dir)
  index_path = os.path.join(report_dir, 'index.html')
  outfile =  open(index_path, 'w') 
  ds_report_dir_abspath = os.path.join(report_dir, ds_report_dir)
  if not os.path.exists(ds_report_dir_abspath):
    os.mkdir(ds_report_dir_abspath)

  store_report(store, outfile, ds_report_dir)

