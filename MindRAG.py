# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 17:14:27 2024

@author: karle
"""

from datetime import datetime, timedelta
from dateutil.parser import parse
from nltk.translate.bleu_score import sentence_bleu
import uuid
from semantic_router.encoders import OpenAIEncoder
from getpass import getpass
import numpy as np
import os, sys
cwd = os.getcwd()
file_path = r"C:\Users\karle\OneDrive - ltu.se\Dokument\PHD\Langchain_KnowIT\GPTech_streamlit.py"
sys.path.append(os.path.join(os.path.dirname(file_path)))
#print(sys.path)
#import QA_doc_agent
#import GPTech
#import LLCMM
import json
from datetime import datetime, timedelta
import LLCMM_support_functions
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
nap_file_name = "pm1_nap_list.json"
with open(nap_file_name, "r") as outfile:
    pm1_nap_list = json.load(outfile)
    
os.environ["OPENAI_API_KEY"] = [YOUR KEY]

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("OpenAI API key: ")

encoder = OpenAIEncoder(name="text-embedding-3-small")


def create_CM_vector_store(training_data):
    '''
    Takes a CM dataset and generates a vector store list/json that can be RAGed using either signal or language properties, and filtered along any
    dimensionality of CM data
    '''
    documents = {}
    documents = compute_vectors(documents, training_data)
            
    return documents

def update_vector_store(documents, training_data):
    # load vector store
    #documents = np.load('doc_file_name.npy', allow_pickle = True) # doc_file_name
    documents = compute_vectors(documents, training_data)
    return documents
    
def compute_vectors(documents, training_data):
    for asset_slice, asset, asset_points, asset_note in training_data:
   
        asset_path = " ".join(asset['Path'].split("\\")[2:])
        asset_name = asset_path
        
        note_comment = asset_note['noteComment']
        note_embedding = encoder([note_comment])
        notedate = asset_note['noteDateUTC']
        idNote = asset_note['idNote']
        notedate = datetime.strptime(notedate[0:19], '%Y-%m-%dT%H:%M:%S')
        
        for point_slice_nr, (point, point_slice) in enumerate(zip(asset_points, asset_slice)):
            recording_chunks = {}
            point_name = point['Name']
            point_id = point['ID']
            for recording_nr, recording_slice in enumerate(point_slice):
                recording = recording_slice[0] #potential to use multiple recordings per representation
                TS = recording[0]
                spectra = recording[1]
                levels = recording[2]
                bias = recording[3]
                dates = recording[4]
                speeds = recording[5]
                timedelta = recording[6]
    
                signal_embedding = normalise_speed(spectra, speeds)
                metadata = { # not necessary to include point level data as recording metadata
                    # "asset": asset,
                    # "point": point,
                    # "annotation": asset_note,
                    # "note_embedding": note_embedding,
                    "TS": TS,
                    "spectra": spectra,
                    "levels": levels,
                    "bias": bias,
                    "dates": dates,
                    "speeds": speeds,
                    "timedelta": timedelta,
                    
                }
                chunk_id = str(uuid.uuid4())
                recording_chunks[chunk_id] = {"signal_embedding": signal_embedding, "metadata": metadata}
            point_chunks = {"note_embedding": note_embedding, "note": asset_note, "asset": asset, "point": point, "recording_chunks": recording_chunks}
            doc_id = (point_id, idNote)
            documents[doc_id] = point_chunks
    return documents

def normalise_speed(spectra, speed):
    '''
    Normalises spectra to same order representation.
    Orinigal data has 500 Hz over 3200 data points, but is sampled from a reality with different shaft speeds.
    By upsampling the signals, recordings with lower shaft speeds now instead have longer vectors to indicate the higher nr of orders possible
    '''
    from scipy import signal
    '''
    Computation to get speed scaling value:
    get_speeds(documents)
    max_speed = max(speeds) #1708
    min_speed = min(speeds)
    recording_length = 3200
    max_resolution = 3200 * max_speed / min_speed # = 12 921
    resample_target = max_resolution * min_speed / speed
    simplified_formula = 3200 * max_speed / speed 
    pre_compute 3200*max_speed and divide by speed when apt
    3200*max_speed = 5 465 600
    '''
    resample_factor = int(5465600/speed)
    resampled_spectra = signal.resample(spectra, resample_factor, domain = 'time')
    return resampled_spectra

def get_notes(documents):
    notes = []
    for doc_id, doc in documents.items():
        notes.append(doc['note'])
    return notes

def get_points(documents):
    points = []
    for doc_id, doc in documents.items():
        points.append(doc['point'])
    return points

def get_assets(documents):
    assets = []
    for doc_id, doc in documents.items():
        assets.append(doc['asset'])
    return assets

def get_speeds(documents):
    speeds = []
    for doc_id, doc in documents.items():
        for chunk_id, recording_chunk in doc['recording_chunks'].items():
            speeds.append(recording_chunk['metadata']['speeds'])
          
    return speeds

def compute_path_distance(document_1, document_2):
    '''
    Computes and returns the distance between two assets in the hierarchy, 
    where distance is defined as number of steps required to go between the assets in the hierarchy.
    '''

    asset_1 = document_1['asset']
    path_1 = asset_1['Path'].split('\\')
    path_1_len = len(path_1)
    asset_2 = document_2['asset']
    path_2 = asset_2['Path'].split('\\')
    path_2_len = len(path_2)

    for path_1_steps in range(path_1_len):
        for path_2_steps in range(path_2_len):
            if path_1[0:path_1_len-path_1_steps] == path_2[0:path_2_len-path_2_steps]:
                nr_of_steps = path_1_steps + path_2_steps
                break
        if path_1[0:path_1_len-path_1_steps] == path_2[0:path_2_len-path_2_steps]:
            break
    return nr_of_steps
        
def compute_bleu_score(point_1, point_2):
    from nltk.translate.bleu_score import sentence_bleu
    # point_1 = document_1['point']
    point_1_name = point_1['Name'].split(" ")
    point_1_type = point_1['DetectionName']

    point_2_name = point_2['Name'].split(" ")
    point_2_type = point_2['DetectionName']

    if point_1_type != point_2_type:
        print("Not same type")

    print(point_1_name)
    print(point_2_name)
    
    bleu_score = sentence_bleu([point_1_name], point_2_name, weights=(1, 0, 0, 0))
    return bleu_score

def compute_note_embedding_distance(vector_store, query_str, top_k):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.
    """
    # Get the embedding for the query string
    query_str_embedding = np.array(encoder(query_str)[0])
    norm_query = np.linalg.norm(query_str_embedding)
    scores = {}
    
    ['note_embedding', 'note', 'asset', 'point', 'recording_chunks']

    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    for doc_id, doc in vector_store.items():
        embedding_array = np.array(doc['note_embedding'][0])
        norm_chunk = np.linalg.norm(embedding_array)
        if norm_query == 0 or norm_chunk == 0:
            # Avoid division by zero
            score = 0
        else:
            score = np.dot(embedding_array, query_str_embedding) / (norm_query * norm_chunk)
        chunk_id = 0
        scores[(doc_id, chunk_id)] = score
        
    # Sort scores and return the top_k results
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_results = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]

    return top_results

def spectrum_encoder(spectrum):
    return spectrum

def compute_spectrum_embedding_distance(vector_store, query_spectrum, top_k, source_asset = False, source_point = False, source_note = False):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.

    It would be good if the source for the spectrum could be included.
    What is query_spectrum is either a naked signal, or a signal including point information?
    Could add source_point to facilitate inclusion or exclusion of source properties.
    Let's add source_notes as placeholder as well.
    """
    # Get the embedding for the query string
    
    query_spectrum_embedding = np.array(spectrum_encoder(query_spectrum))
    norm_query = np.linalg.norm(query_spectrum_embedding)
    scores = {}
    
    ['note_embedding', 'note', 'asset', 'point', 'recording_chunks']

    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    for doc_id, doc in vector_store.items():
        for chunk_id, recording_chunk in doc['recording_chunks'].items():
            embedding_array = np.array(recording_chunk['signal_embedding'][0:3200])
            norm_chunk = np.linalg.norm(embedding_array)
            if norm_query == 0 or norm_chunk == 0:
                # Avoid division by zero
                score = 0
            else:
                score = np.dot(embedding_array, query_spectrum_embedding) / (norm_query * norm_chunk)

            scores[(doc_id, chunk_id)] = score
        
    # Sort scores and return the top_k results
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_results = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]

    return top_results

def compute_path_short(path_1, path_2):
    '''
    Computes and returns the distance between two assets in the hierarchy, 
    where distance is defined as number of steps required to go between the assets in the hierarchy.
    '''
    path_1 = path_1.split('\\')
    path_1_len = len(path_1)
    path_2 = path_2.split('\\')
    path_2_len = len(path_2)

    for path_1_steps in range(path_1_len):
        for path_2_steps in range(path_2_len):
            if path_1[0:path_1_len-path_1_steps] == path_2[0:path_2_len-path_2_steps]:
                nr_of_steps = path_1_steps + path_2_steps
                break
        if path_1[0:path_1_len-path_1_steps] == path_2[0:path_2_len-path_2_steps]:
            break
    asset_bleu_score = sentence_bleu([path_1], path_2, weights=(1, 0, 0, 0))

    return nr_of_steps, asset_bleu_score

def compute_bleu_short(point_1, point_2):
    point_1_name = point_1['Name'].split(" ")
    point_1_type = point_1['DetectionName']

    point_2_name = point_2['Name'].split(" ")
    point_2_type = point_2['DetectionName']

    bleu_score = sentence_bleu([point_1_name], point_2_name, weights=(1, 0, 0, 0))
    detection_name_sim = False
    if point_1_type == point_2_type:
        detection_name_sim = True # we're treating points with different signal types as completely different
    return bleu_score, detection_name_sim
        
def compute_score(query_embedding, norm_query, embedding_array):
    norm_chunk = np.linalg.norm(embedding_array)
    if norm_query == 0 or norm_chunk == 0:
        # Avoid division by zero
        score = 0
    else:
        score = np.dot(embedding_array, query_embedding) / (norm_query * norm_chunk)
    return score

def compute_recording_scores(recording_1, recording_2, conditions = False):
    '''
    Will split into multiple funcs that take in metadata?
    Should be remade to take one recording and the doc.
    Makes compute score easier to use as well.
    Sort each score individually or sort them together?
    '''
    metadata_1 = recording_1['metadata']
    metadata_2 = recording_2['metadata']
    
    ts_1 = metadata_1['TS']
    ts_2 = metadata_2['TS']
    ts_score = compute_score(ts_1, np.linalg.norm(ts_1), ts_2)

    trend_1 = metadata_1['levels']
    trend_2 = metadata_2['levels']
    # compare the trends
    trends_score = compute_score(trend_1, np.linalg.norm(trend_1), trend_2)
    
    bias_1 = metadata_1['bias']
    bias_2 = metadata_2['bias']    
    # compare the variance in the trend_bias
    epsilon = 1 #1e-6
    # bias_overlap = np.abs(bias_1-bias_2)
    bias_score = 1/(np.square(np.var(bias_1)-np.var(bias_2))+epsilon)

    # check how close the two recordings are
    recording_criteria = timedelta(5)
    dates_1 = parse(metadata_1['dates'])
    dates_2 = parse(metadata_2['dates'])
    recording_td = dates_1 - dates_2
    is_in_td = False
    if recording_td<recording_criteria:
        is_in_td = True
    recording_td = 1/(np.abs(recording_td.total_seconds())+1)
    
    # check how close the two recordings are to their respective annotation
    # this can be relevant if annotation critera are also considered, so that it's easy to check for 
    # "signals with similar annotations that are also at a similar time before fault"
    td_criteria = timedelta(5)
    timedelta_1 = metadata_1['timedelta']
    timedelta_2 = metadata_2['timedelta']
    td_td = timedelta_1 - timedelta_2
    is_in_td_td = False
    if td_td<td_criteria:
        is_in_td_td = True
    td_td = 1/(np.abs(td_td.total_seconds())+1)

    # check if the recordings are operating at the same speeds
    speeds_1 = metadata_1['speeds']
    speeds_2 = metadata_2['speeds']    
    # compare the variance in the trend_bias
    epsilon = 1e-6
    epsilon = 1
    speed_overlap = np.abs(speeds_1-speeds_2)
    speed_score = 1/(np.square(speeds_1-speeds_2)+epsilon)
    ###########################################################################################################################
    # Compute features that are relevant to compare between recordings, such as:
    # sensor bias variance
    # level max
    # level rise? (i.e. a "kernel" reacting to if later elements are much larger than smaller elements)
    # anything in TS? hardly relevant
    # points with similar speeds? Can be a relevant query
    # should dates or timedelta be relevant?
    # Could query for other recordings at the same date, or within a span around the date
    # Could query for other recordings with similar timedelta to notes with certain properties
    ###########################################################################################################################

    return trends_score, bias_score, recording_td, td_td, speed_score, ts_score
    
def compute_chunk_embedding_distance(vector_store, query_chunk, top_k, 
                                     input_asset = False, input_point = False, input_note = False, 
                                     weights = [6, 3, 1, 0.5, 0.5, 0.0005], asset_weights = [1, 1, 1],
                                     asset_conditions = [], point_conditions = [], note_conditions = [], recording_conditions = []):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.

    It would be good if the source for the spectrum could be included.
    What is query_spectrum is either a naked signal, or a signal including point information?
    Could add source_point to facilitate inclusion or exclusion of source properties.
    Let's add source_notes as placeholder as well.
    """
    # Get the embedding for the query string
   # query_spectrum = query_chunk['signal_embedding']
    ###
    # Here check if the chunk contains metadata, and if it should be applied
    # Call compute recording sims with chunk and doc
    #
    ###
    epsilon = 1e-6
    metadata = query_chunk['metadata']
    # asset = metadata['asset']
    # point = metadata['point']
    # annotation = metadata['annotation']
    # note_embedding = metadata['note_embedding']
    query_spectra = metadata['spectra']
    signal_embedding = query_chunk['signal_embedding']
    query_spectrum_embedding = signal_embedding #np.array(spectrum_encoder(query_spectrum))
    norm_spectrum_query = np.linalg.norm(query_spectrum_embedding)
    raw_norm_spectrum_query = np.linalg.norm(query_spectra)
    if any(input_note):
        query_note_embedding = np.array(encoder(input_note)[0])
        norm_note_query = np.linalg.norm(query_note_embedding)
        concat_query_embedding = np.concatenate((query_spectrum_embedding[0:3200], query_note_embedding))
        concat_query_norm = np.linalg.norm(concat_query_embedding)
    

    note_scores = {}
    point_scores = {}
    asset_scores = {}
    
    scores = {}
    concat_scores = {}
    trends_scores = {}
    bias_scores = {}
    recording_tds = {}
    td_tds = {}
    speed_overlaps = {}
    ts_scores = {}
    raw_spectrum_scores = {}
    
    # ['note_embedding', 'note', 'asset', 'point', 'recording_chunks']

    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    for doc_id, doc in vector_store.items():
        ####################################################################################
        # Compute similarity for annotations
        # Compute similarity for point properties
        # Compute similarity for asset properties x
        # Compute similarity for recording properties
        #
        # A LLM to analyse the returned data to provide fault diagnosis
        # Test how well it analyses spectra with good technical prompts
        # Can send in naked spectra, but also add more data e.g. point, prior notes, etc. , which LLM uses to enhance the diagnoses
        #
        # Retrieve closest data given the input data and distance metric evaluations that are provided, with a "standard" approach coded
        ####################################################################################

        asset = doc['asset']
        point = doc['point']
        note = doc['note']
        note_embedding = doc['note_embedding']
        
        if any(input_asset):
            # compute_path_distance
            path_score, asset_bleu = compute_path_short(input_asset['Path'], asset['Path'])
            asset_scores[doc_id] = path_score
            
        if any(input_point):
            # compute_bleu_distance
            point_score, detection_name_sim = compute_bleu_short(input_point, point)
            point_scores[doc_id] = point_score
            
        if any(input_note):
            # compute embedding_dot_product                       
            note_score = compute_score(query_note_embedding, norm_note_query, np.array(note_embedding[0]))
            input_note_time = parse(input_note['noteDateUTC'])
            note_time = parse(note['noteDateUTC'])
            note_time_diff = input_note_time - note_time
            note_date_sim = timedelta(100)-note_time_diff #1/(np.square(note_time_diff)+epsilon)
            note_scores[doc_id] = note_score

            
        if 'recording_chunks' in doc.keys():
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                                
                embedding_array = np.array(recording_chunk['signal_embedding'][0:3200])
                metadata = query_chunk['metadata']
                spectra = metadata['spectra']
                # norm_chunk = np.linalg.norm(embedding_array)
                
                # recording_span = [200:3200]
                spectrum_score = compute_score(query_spectrum_embedding[0:3200], norm_spectrum_query, embedding_array[0:3200])
                trends_score, bias_score, recording_td, td_td, speed_overlap, ts_score = compute_recording_scores(query_chunk, recording_chunk, conditions = False)
                raw_spectrum_score = compute_score(query_spectra[0:3200], raw_norm_spectrum_query, spectra[0:3200])
                scores[(doc_id, chunk_id)] = spectrum_score
                trends_scores[(doc_id, chunk_id)] = trends_score
                bias_scores[(doc_id, chunk_id)] = bias_score
                recording_tds[(doc_id, chunk_id)] = recording_td
                td_tds[(doc_id, chunk_id)] = td_td
                speed_overlaps[(doc_id, chunk_id)] = speed_overlap
                ts_scores[(doc_id, chunk_id)] = ts_score
                raw_spectrum_scores[(doc_id, chunk_id)] = raw_spectrum_score

                # compute_score with both spectrum embedding and note embedding jointly
                if any(input_note):
                    concat_embedding = np.concatenate((embedding_array[0:3200], np.array(note_embedding[0])))
                    concat_score = compute_score(concat_query_embedding, concat_query_norm, concat_embedding)
                    concat_scores[(doc_id, chunk_id)] = concat_score
    
    # Sort scores and return the top_k results
    top_results = sort_scores(scores, top_k)
    top_results_trends = sort_scores(trends_scores, top_k)
    top_results_bias = sort_scores(bias_scores, top_k)
    top_results_recording_tds = sort_scores(recording_tds, top_k)
    top_results_tds = sort_scores(td_tds, top_k)
    top_results_speeds = sort_scores(speed_overlaps, top_k)
    top_results_ts = sort_scores(ts_scores, top_k)
    top_results_sp = sort_scores(raw_spectrum_scores, top_k)
    
    avg_scores = summarise_scores(scores, trends_scores, bias_scores, recording_tds, td_tds, speed_overlaps, weights = weights)
    top_results_avg = sort_scores(avg_scores, top_k)

    top_results_concat = sort_scores(concat_scores, top_k)
    
    top_results_assets = sort_asset_level_scores(asset_scores, top_k)
    top_results_points = sort_asset_level_scores(point_scores, top_k)
    top_results_notes = sort_asset_level_scores(note_scores, top_k)

    avg_scores_hierarchy = summarise_hierarchy_scores(asset_scores, point_scores, note_scores, asset_weights = asset_weights)
    top_results_hierarchy = sort_asset_level_scores(avg_scores_hierarchy, top_k)
    
    # return top_results, top_results_trends, top_results_bias, top_results_recording_tds, top_results_tds, top_results_speeds, top_results_avg, top_results_assets, top_results_points, top_results_notes, top_results_hierarchy, top_results_concat
    return top_results, top_results_trends, top_results_bias, top_results_recording_tds, top_results_tds, top_results_speeds, top_results_avg, top_results_assets, top_results_points, top_results_notes, top_results_hierarchy, top_results_concat, top_results_ts, top_results_sp

def compute_recordings_embedding_distance(vector_store, query_chunks, top_k):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.

    It would be good if the source for the spectrum could be included.
    What is query_spectrum is either a naked signal, or a signal including point information?
    Could add source_point to facilitate inclusion or exclusion of source properties.
    Let's add source_notes as placeholder as well.
    """
    # Get the embedding for the query string
   # query_spectrum = query_chunk['signal_embedding']
    ###
    # Here check if the chunk contains metadata, and if it should be applied
    # Call compute recording sims with chunk and doc
    #
    ###
    
    # query_top_results_emb, query_top_results_sp, query_top_results_ts = [], [], []
    
    np_query_spectra = np.zeros((len(query_chunks), 3200))
    np_query_ts = np.zeros((len(query_chunks), 8192))
    np_query_spectrum_embedding = np.zeros((len(query_chunks), 3200))
    np_norm_spectrum_query = np.zeros((len(query_chunks), 1))
    np_raw_norm_spectrum_query = np.zeros((len(query_chunks), 1))
    np_norm_ts_query = np.zeros((len(query_chunks), 1))

    for query_chunk_nr, query_chunk in enumerate(query_chunks):
        metadata = query_chunk['metadata']
        # asset = metadata['asset']
        # point = metadata['point']
        # annotation = metadata['annotation']
        # note_embedding = metadata['note_embedding']
        query_spectra = np.array(metadata['spectra'][0:3200])
        np_query_spectra[query_chunk_nr]=query_spectra
        query_ts = np.array(metadata['TS'][0:8192])
        np_query_ts[query_chunk_nr]=query_ts
        signal_embedding = query_chunk['signal_embedding']
        query_spectrum_embedding = np.array(signal_embedding[0:3200]) #np.array(spectrum_encoder(query_spectrum))
        np_query_spectrum_embedding[query_chunk_nr]=query_spectrum_embedding
        
        norm_spectrum_query = np.linalg.norm(query_spectrum_embedding)
        np_norm_spectrum_query[query_chunk_nr]=norm_spectrum_query
        raw_norm_spectrum_query = np.linalg.norm(query_spectra)
        np_raw_norm_spectrum_query[query_chunk_nr]=raw_norm_spectrum_query
        norm_ts_query = np.linalg.norm(query_ts)
        np_norm_ts_query[query_chunk_nr]=norm_ts_query
    
    emb_scores = [{}]*len(query_chunks)
    time_series_scores = [{}]*len(query_chunks)
    raw_spectrum_scores = [{}]*len(query_chunks)
    
    # ['note_embedding', 'note', 'asset', 'point', 'recording_chunks']
    
    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    
    for doc_id, doc in vector_store.items():
        ####################################################################################
        # Compute similarity for annotations
        # Compute similarity for point properties
        # Compute similarity for asset properties x
        # Compute similarity for recording properties
        #
        # Ask LLM to analyse the returned data to provide fault diagnosis
        # Test how well it analyses spectra with good technical prompts
        # Can send in naked spectra, but also add more data e.g. point, prior notes, etc. , which LLM uses to enhance the diagnoses
        #
        # Retrieve closest data given the input data and distance metric evaluations that are provided, with a "standard" approach coded
        ####################################################################################
        doc_embeddings = np.zeros((len(doc['recording_chunks']), 3200))
        doc_spectra = np.zeros((len(doc['recording_chunks']), 3200))
        doc_ts = np.zeros((len(doc['recording_chunks']), 8192))
            
        if 'recording_chunks' in doc.keys():
            for chunk_nr, (chunk_id, recording_chunk) in enumerate(doc['recording_chunks'].items()):
                
                if len(recording_chunk['signal_embedding']) < 3200:
                    recording_chunk['signal_embedding'] = np.pad(recording_chunk['signal_embedding'], (0, 3200-len(recording_chunk['signal_embedding'])), 'constant', constant_values=(0, 0))                    
 
                embedding_array = np.array(recording_chunk['signal_embedding'][0:3200])
                
                doc_embeddings[chunk_nr] = embedding_array
                
                metadata = recording_chunk['metadata']
                spectra = np.array(metadata['spectra'][0:3200])
                doc_spectra[chunk_nr] = spectra
                time_series = np.array(metadata['TS'][0:8192])
                doc_ts[chunk_nr] = time_series
        
            embedding_norms = np.linalg.norm(doc_embeddings, axis = 1)
            embedding_scores = np.matmul(doc_embeddings, np.transpose(np_query_spectrum_embedding)) / np.transpose(np_norm_spectrum_query * embedding_norms + 1e-6)
            spectrum_norms = np.linalg.norm(doc_spectra, axis = 1)
            spectrum_scores = np.matmul(doc_spectra, np.transpose(np_query_spectra)) / np.transpose(np_raw_norm_spectrum_query * spectrum_norms + 1e-6)
            ts_norms = np.linalg.norm(doc_ts, axis = 1)
            ts_scores = np.matmul(doc_ts, np.transpose(np_query_ts)) / np.transpose(np_norm_ts_query * ts_norms + 1e-6)
            for query_chunk_nr in range(len(query_chunks)):
                for chunk_nr, (chunk_id) in enumerate(doc['recording_chunks'].keys()):
                    emb_scores[query_chunk_nr][(doc_id, chunk_id)] = embedding_scores[chunk_nr, query_chunk_nr]
                    time_series_scores[query_chunk_nr][(doc_id, chunk_id)] = ts_scores[chunk_nr, query_chunk_nr]
                    raw_spectrum_scores[query_chunk_nr][(doc_id, chunk_id)] = spectrum_scores[chunk_nr, query_chunk_nr]
    '''
    top_results = sort_scores(scores, top_k)
    query_top_results_emb.append(top_results)
    top_results_ts = sort_scores(time_series_scores, top_k)
    query_top_results_ts.append(top_results_ts)
    top_results_sp = sort_scores(raw_spectrum_scores, top_k)
    query_top_results_sp.append(top_results_sp)
    # Sort scores and return the top_k results
   
    top_results = sort_scores(scores, top_k)
    top_results_ts = sort_scores(time_series_scores, top_k)
    top_results_sp = sort_scores(raw_spectrum_scores, top_k)
    '''
    
    # return top_results, top_results_trends, top_results_bias, top_results_recording_tds, top_results_tds, top_results_speeds, top_results_avg, top_results_assets, top_results_points, top_results_notes, top_results_hierarchy, top_results_concat
    return emb_scores, time_series_scores, raw_spectrum_scores

import copy
def compute_recordings_id_embedding_distance(vector_store, query_chunk_keys, query_chunks = [], chunk_doc_id = 0, exclude_same_point = True):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.

    It would be good if the source for the spectrum could be included.
    What is query_spectrum is either a naked signal, or a signal including point information?
    Could add source_point to facilitate inclusion or exclusion of source properties.
    Let's add source_notes as placeholder as well.
    
    Query chunks are now the actual chunks, i.e. a recording from the doc
    """
    # Get the embedding for the query string
   # query_spectrum = query_chunk['signal_embedding']
    ###
    # Here check if the chunk contains metadata, and if it should be applied
    # Call compute recording sims with chunk and doc
    #
    ###
    if not any(query_chunks):
        query_chunks = get_chunks_from_chunk_keys(vector_store, query_chunk_keys)
    if not chunk_doc_id:
        chunk_doc_id, _ = get_keys_from_chunk_keys(vector_store, query_chunk_keys[0])
    # query_top_results_emb, query_top_results_sp, query_top_results_ts = [], [], []
    
    # np_query_spectra = np.zeros((len(query_chunks), 3200))
    # np_query_ts = np.zeros((len(query_chunks), 8192))
    
    np_query_spectrum_embedding = np.zeros((len(query_chunks), 3200))
    np_norm_spectrum_query = np.zeros((len(query_chunks), 1))
    # np_raw_norm_spectrum_query = np.zeros((len(query_chunks), 1))
    # np_norm_ts_query = np.zeros((len(query_chunks), 1))
    query_emb_scores = {}
    for query_chunk_nr, (query_chunk_id, query_chunk) in enumerate(query_chunks.items()):
    # for query_chunk_nr, query_chunk in enumerate(query_chunks):
        # metadata = query_chunk['metadata']
        # asset = metadata['asset']
        # point = metadata['point']
        # annotation = metadata['annotation']
        # note_embedding = metadata['note_embedding']
        # query_spectra = np.array(metadata['spectra'][0:3200])
        # np_query_spectra[query_chunk_nr]=query_spectra
        # query_ts = np.array(metadata['TS'][0:8192])
        # np_query_ts[query_chunk_nr]=query_ts
        signal_embedding = query_chunk['signal_embedding']
        query_spectrum_embedding = np.zeros(3200)
        query_spectrum_embedding[0:len(signal_embedding[0:3200])] = np.array(signal_embedding[0:3200]) #np.array(spectrum_encoder(query_spectrum))
        np_query_spectrum_embedding[query_chunk_nr]=query_spectrum_embedding
        
        norm_spectrum_query = np.linalg.norm(query_spectrum_embedding)
        np_norm_spectrum_query[query_chunk_nr]=norm_spectrum_query
        # raw_norm_spectrum_query = np.linalg.norm(query_spectra)
        # np_raw_norm_spectrum_query[query_chunk_nr]=raw_norm_spectrum_query
        # norm_ts_query = np.linalg.norm(query_ts)
        # np_norm_ts_query[query_chunk_nr]=norm_ts_query
        query_emb_scores[query_chunk_id] = {}
    # emb_scores = [{} for i in range(len(query_chunks))]
    
    
    # time_series_scores = [{}]*len(query_chunks)
    # raw_spectrum_scores = [{}]*len(query_chunks)
    
    # ['note_embedding', 'note', 'asset', 'point', 'recording_chunks']
    
    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    asset_point_labels = []
    asset_notes_labels = []
    for doc_id, doc in vector_store.items():
        if exclude_same_point and doc_id == chunk_doc_id:
            query_asset = doc['asset']
            query_points = doc['point']
            pointname = query_points['Name']
            point_label_value = make_point_label(pointname)
            point_label_values = [point_label_value]*len(query_chunks)
            asset_point_labels.extend(point_label_values)
            
            query_note = doc['note']
            query_notecontent = query_note['noteComment']
            note_label_value = make_note_label(query_notecontent)
            note_label_values = [note_label_value]*len(query_chunks)
            asset_notes_labels.extend(note_label_values)

            continue 
        ####################################################################################
        # Compute similarity for annotations
        # Compute similarity for point properties
        # Compute similarity for asset properties x
        # Compute similarity for recording properties
        #
        # Ask LLM to analyse the returned data to provide fault diagnosis
        # Test how well it analyses spectra with good technical prompts
        # Can send in naked spectra, but also add more data e.g. point, prior notes, etc. , which LLM uses to enhance the diagnoses
        #
        # Retrieve closest data given the input data and distance metric evaluations that are provided, with a "standard" approach coded
        ####################################################################################
        recording_chunks=doc['recording_chunks']
        doc_embeddings = np.zeros((len(recording_chunks), 3200))
        # doc_spectra = np.zeros((len(doc['recording_chunks']), 3200))
        # doc_ts = np.zeros((len(doc['recording_chunks']), 8192))
            

        for chunk_nr, (chunk_id, recording_chunk) in enumerate(recording_chunks.items()):
            
            if len(recording_chunk['signal_embedding']) < 3200:
                recording_chunk['signal_embedding'] = np.pad(recording_chunk['signal_embedding'], (0, 3200-len(recording_chunk['signal_embedding'])), 'constant', constant_values=(0, 0))                    
 
            embedding_array = np.array(recording_chunk['signal_embedding'][0:3200])
            
            doc_embeddings[chunk_nr] = embedding_array
            
            # metadata = recording_chunk['metadata']
            # spectra = np.array(metadata['spectra'][0:3200])
            # doc_spectra[chunk_nr] = spectra
            # time_series = np.array(metadata['TS'][0:8192])
            # doc_ts[chunk_nr] = time_series
    
        embedding_norms = np.linalg.norm(doc_embeddings, axis = 1)
        embedding_scores = np.matmul(doc_embeddings, np.transpose(np_query_spectrum_embedding)) / np.transpose(np_norm_spectrum_query * embedding_norms + 1e-6)
        embedding_scores_t = np.transpose(embedding_scores)
        # spectrum_norms = np.linalg.norm(doc_spectra, axis = 1)
        # spectrum_scores = np.matmul(doc_spectra, np.transpose(np_query_spectra)) / np.transpose(np_raw_norm_spectrum_query * spectrum_norms + 1e-6)
        # ts_norms = np.linalg.norm(doc_ts, axis = 1)
        # ts_scores = np.matmul(doc_ts, np.transpose(np_query_ts)) / np.transpose(np_norm_ts_query * ts_norms + 1e-6)
        
        for query_chunk_nr, query_chunk_key in enumerate(query_chunk_keys):
            # emb_scores = {}
            for chunk_nr, chunk_id in enumerate(recording_chunks.keys()):
                # emb_scores[(doc_id, chunk_id)] = embedding_scores_t[query_chunk_nr, chunk_nr]
                query_emb_scores[query_chunk_key][(doc_id, chunk_id)] = embedding_scores_t[query_chunk_nr, chunk_nr]
                # = copy.deepcopy(emb_scores)
                
        # for chunk_nr, chunk_id in enumerate(recording_chunks.keys()):
        #     for query_chunk_nr, query_chunk_key in enumerate(query_chunk_keys):
        #         emb_scores[query_chunk_nr][(doc_id, chunk_id)] = embedding_scores[chunk_nr, query_chunk_nr]
                # time_series_scores[query_chunk_nr][(doc_id, chunk_id)] = ts_scores[chunk_nr, query_chunk_nr]
                # raw_spectrum_scores[query_chunk_nr][(doc_id, chunk_id)] = spectrum_scores[chunk_nr, query_chunk_nr]
                
                    
    '''
    top_results = sort_scores(scores, top_k)
    query_top_results_emb.append(top_results)
    top_results_ts = sort_scores(time_series_scores, top_k)
    query_top_results_ts.append(top_results_ts)
    top_results_sp = sort_scores(raw_spectrum_scores, top_k)
    query_top_results_sp.append(top_results_sp)
    # Sort scores and return the top_k results
   
    top_results = sort_scores(scores, top_k)
    top_results_ts = sort_scores(time_series_scores, top_k)
    top_results_sp = sort_scores(raw_spectrum_scores, top_k)
    '''

    # query_emb_scores = {}
    # for query_chunk_nr, query_chunk_key in enumerate(query_chunk_keys):
    #     query_emb_scores[query_chunk_key] = emb_scores[query_chunk_nr]
    # return top_results, top_results_trends, top_results_bias, top_results_recording_tds, top_results_tds, top_results_speeds, top_results_avg, top_results_assets, top_results_points, top_results_notes, top_results_hierarchy, top_results_concat
    return query_emb_scores, asset_point_labels, asset_notes_labels

def compute_embedding_distance_final(vector_store, query_chunk_keys, query_chunks = [], chunk_doc_id = 0, exclude_same_point = True):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.

    It would be good if the source for the spectrum could be included.
    What is query_spectrum is either a naked signal, or a signal including point information?
    Could add source_point to facilitate inclusion or exclusion of source properties.
    Let's add source_notes as placeholder as well.
    
    Query chunks are now the actual chunks, i.e. a recording from the doc
    """
    # Get the embedding for the query string
   # query_spectrum = query_chunk['signal_embedding']
    ###
    # Here check if the chunk contains metadata, and if it should be applied
    # Call compute recording sims with chunk and doc
    #
    ###
    if not any(query_chunks):
        query_chunks = get_chunks_from_chunk_keys(vector_store, query_chunk_keys)
    if not chunk_doc_id:
        chunk_doc_id, _ = get_keys_from_chunk_keys(vector_store, query_chunk_keys[0])
    # query_top_results_emb, query_top_results_sp, query_top_results_ts = [], [], []
    
    query_asset, query_point, query_note = get_hierarchy_from_chunk_key(vector_store, query_chunk_keys[0])
    query_asset_id = query_asset['ID']
    # np_query_spectra = np.zeros((len(query_chunks), 3200))
    # np_query_ts = np.zeros((len(query_chunks), 8192))
    
    np_query_spectrum_embedding = np.zeros((len(query_chunks), 3202))
    np_norm_spectrum_query = np.zeros((len(query_chunks), 1))
    # np_raw_norm_spectrum_query = np.zeros((len(query_chunks), 1))
    # np_norm_ts_query = np.zeros((len(query_chunks), 1))
    query_emb_scores = {}
    for query_chunk_nr, (query_chunk_id, query_chunk) in enumerate(query_chunks.items()):
    # for query_chunk_nr, query_chunk in enumerate(query_chunks):
        metadata = query_chunk['metadata']
        # asset = metadata['asset']
        # point = metadata['point']
        # annotation = metadata['annotation']
        # note_embedding = metadata['note_embedding']
        query_spectra = np.array(metadata['spectra'][0:3200])
        query_spectrum_embedding = query_spectra
        # np_query_spectra[query_chunk_nr]=query_spectra
        # query_ts = np.array(metadata['TS'][0:8192])
        # np_query_ts[query_chunk_nr]=query_ts
        '''
        signal_embedding = query_chunk['signal_embedding']
        query_spectrum_embedding = np.zeros(3200)
        query_spectrum_embedding[0:len(signal_embedding[0:3200])] = np.array(signal_embedding[0:3200]) #np.array(spectrum_encoder(query_spectrum))
        query_spectrum_embedding = query_spectrum_embedding-np.average(query_spectrum_embedding)
        query_spectrum_embedding = np.where(query_spectrum_embedding<0, 0, query_spectrum_embedding)
        '''
        np_query_spectrum_embedding[query_chunk_nr]=np.append(query_spectrum_embedding, [skew(query_spectrum_embedding), kurtosis(query_spectrum_embedding)])
        
        norm_spectrum_query = np.linalg.norm(query_spectrum_embedding)
        np_norm_spectrum_query[query_chunk_nr]=norm_spectrum_query
        # raw_norm_spectrum_query = np.linalg.norm(query_spectra)
        # np_raw_norm_spectrum_query[query_chunk_nr]=raw_norm_spectrum_query
        # norm_ts_query = np.linalg.norm(query_ts)
        # np_norm_ts_query[query_chunk_nr]=norm_ts_query
        query_emb_scores[query_chunk_id] = {}
    # emb_scores = [{} for i in range(len(query_chunks))]
    
    
    # time_series_scores = [{}]*len(query_chunks)
    # raw_spectrum_scores = [{}]*len(query_chunks)
    
    # ['note_embedding', 'note', 'asset', 'point', 'recording_chunks']
    
    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    asset_point_labels = []
    asset_notes_labels = []
    for doc_id, doc in vector_store.items():
        asset = doc['asset']
        asset_id = asset['ID']
        query_points = doc['point']
        if doc_id == chunk_doc_id:
            
            pointname = query_points['Name']
            point_label_value = make_point_label(pointname)
            point_label_values = [point_label_value]*len(query_chunks)
            asset_point_labels.extend(point_label_values)
            
            query_note = doc['note']
            query_notecontent = query_note['noteComment']
            note_label_value = make_note_label(query_notecontent)
            note_label_values = [note_label_value]*len(query_chunks)
            asset_notes_labels.extend(note_label_values)

        if query_asset_id != asset_id or not exclude_same_point:
        ####################################################################################
        # Compute similarity for annotations
        # Compute similarity for point properties
        # Compute similarity for asset properties x
        # Compute similarity for recording properties
        #
        # Ask LLM to analyse the returned data to provide fault diagnosis
        # Test how well it analyses spectra with good technical prompts
        # Can send in naked spectra, but also add more data e.g. point, prior notes, etc. , which LLM uses to enhance the diagnoses
        #
        # Retrieve closest data given the input data and distance metric evaluations that are provided, with a "standard" approach coded
        ####################################################################################
            recording_chunks=doc['recording_chunks']
            doc_embeddings = np.zeros((len(recording_chunks), 3202))
            # doc_spectra = np.zeros((len(doc['recording_chunks']), 3200))
            # doc_ts = np.zeros((len(doc['recording_chunks']), 8192))
                
    
            for chunk_nr, (chunk_id, recording_chunk) in enumerate(recording_chunks.items()):
                
                # if len(recording_chunk['signal_embedding']) < 3200:
                #     recording_chunk['signal_embedding'] = np.pad(recording_chunk['signal_embedding'], (0, 3200-len(recording_chunk['signal_embedding'])), 'constant', constant_values=(0, 0))                    
                
                '''
                target_embedding = np.array(recording_chunk['signal_embedding'])#[0:3200])
                embedding_array = np.zeros(3200)
                embedding_array[0:len(target_embedding[0:3200])] = np.array(target_embedding[0:3200]) #np.array(spectrum_encoder(query_spectrum))
                embedding_array = embedding_array-np.average(embedding_array)
                embedding_array = np.where(embedding_array<0, 0, embedding_array)
                # embedding_array[chunk_nr]=embedding_array
                '''
                
                
                metadata = recording_chunk['metadata']
                spectra = np.array(metadata['spectra'][0:3200])
                embedding_array = spectra
                # doc_spectra[chunk_nr] = spectra
                # time_series = np.array(metadata['TS'][0:8192])
                # doc_ts[chunk_nr] = time_series
                
                doc_embeddings[chunk_nr] = np.append(embedding_array, [skew(embedding_array), kurtosis(embedding_array)])
        
            embedding_norms = np.linalg.norm(doc_embeddings, axis = 1)
            embedding_scores = np.matmul(doc_embeddings, np.transpose(np_query_spectrum_embedding)) / np.transpose(np_norm_spectrum_query * embedding_norms + 1e-6)
            embedding_scores_t = np.transpose(embedding_scores)
            # spectrum_norms = np.linalg.norm(doc_spectra, axis = 1)
            # spectrum_scores = np.matmul(doc_spectra, np.transpose(np_query_spectra)) / np.transpose(np_raw_norm_spectrum_query * spectrum_norms + 1e-6)
            # ts_norms = np.linalg.norm(doc_ts, axis = 1)
            # ts_scores = np.matmul(doc_ts, np.transpose(np_query_ts)) / np.transpose(np_norm_ts_query * ts_norms + 1e-6)
            
            for query_chunk_nr, query_chunk_key in enumerate(query_chunk_keys):
                # emb_scores = {}
                for chunk_nr, chunk_id in enumerate(recording_chunks.keys()):
                    # emb_scores[(doc_id, chunk_id)] = embedding_scores_t[query_chunk_nr, chunk_nr]
                    query_emb_scores[query_chunk_key][(doc_id, chunk_id)] = embedding_scores_t[query_chunk_nr, chunk_nr]
                    # = copy.deepcopy(emb_scores)
                

                
                    
    '''
    top_results = sort_scores(scores, top_k)
    query_top_results_emb.append(top_results)
    top_results_ts = sort_scores(time_series_scores, top_k)
    query_top_results_ts.append(top_results_ts)
    top_results_sp = sort_scores(raw_spectrum_scores, top_k)
    query_top_results_sp.append(top_results_sp)
    # Sort scores and return the top_k results
   
    top_results = sort_scores(scores, top_k)
    top_results_ts = sort_scores(time_series_scores, top_k)
    top_results_sp = sort_scores(raw_spectrum_scores, top_k)
    '''

    # query_emb_scores = {}
    # for query_chunk_nr, query_chunk_key in enumerate(query_chunk_keys):
    #     query_emb_scores[query_chunk_key] = emb_scores[query_chunk_nr]
    # return top_results, top_results_trends, top_results_bias, top_results_recording_tds, top_results_tds, top_results_speeds, top_results_avg, top_results_assets, top_results_points, top_results_notes, top_results_hierarchy, top_results_concat
    return query_emb_scores, asset_point_labels, asset_notes_labels

def compute_chunks_embedding_distance(vector_store, query_chunks, top_k, 
                                     input_asset = False, input_point = False, input_note = False, 
                                     weights = [6, 3, 1, 0.5, 0.5, 0.0005], asset_weights = [1, 1, 1],
                                     asset_conditions = [], point_conditions = [], note_conditions = [], recording_conditions = []):
    """
    This function takes in a vector store dictionary, a query string, and an int 'top_k'.
    It computes embeddings for the query string and then calculates the cosine similarity against every chunk embedding in the dictionary.
    The top_k matches are returned based on the highest similarity scores.

    It would be good if the source for the spectrum could be included.
    What is query_spectrum is either a naked signal, or a signal including point information?
    Could add source_point to facilitate inclusion or exclusion of source properties.
    Let's add source_notes as placeholder as well.
    """
    # Get the embedding for the query string
   # query_spectrum = query_chunk['signal_embedding']
    ###
    # Here check if the chunk contains metadata, and if it should be applied
    # Call compute recording sims with chunk and doc
    #
    ###
    epsilon = 1e-6
    query_spectrum_embeddings = []
    norm_spectrum_queries = []
    raw_norm_spectrum_queries = []
    concat_query_embeddings = []
    concat_query_norms = []
    if any(input_note):
        if 'noteComment' in input_note.keys():
            if any(input_note['noteComment']):
                input_note_text = [input_note['noteComment']]
                query_note_embedding = np.array(encoder(input_note_text)[0])
                norm_note_query = np.linalg.norm(query_note_embedding)
            else:
                query_note_embedding = 0
                norm_note_query = 0
    for query_chunk in query_chunks:
        metadata = query_chunk['metadata']
        query_spectra = metadata['spectra']
        signal_embedding = query_chunk['signal_embedding']
        query_spectrum_embedding = signal_embedding #np.array(spectrum_encoder(query_spectrum))
        query_spectrum_embeddings.append(query_spectrum_embedding)
        norm_spectrum_query = np.linalg.norm(query_spectrum_embedding)
        norm_spectrum_queries.append(norm_spectrum_query)
        raw_norm_spectrum_query = np.linalg.norm(query_spectra)
        raw_norm_spectrum_queries.append(raw_norm_spectrum_query)
    
        if any(input_note):
            concat_query_embedding = np.concatenate((query_spectrum_embedding[0:3200], query_note_embedding))
            concat_query_embeddings.append(concat_query_embedding)
            concat_query_norm = np.linalg.norm(concat_query_embedding)
            concat_query_norms.append(concat_query_norm)

    note_scores = {}
    point_scores = {}
    asset_scores = {}
    
    scores = {}
    concat_scores = {}
    trends_scores = {}
    bias_scores = {}
    recording_tds = {}
    td_tds = {}
    speed_overlaps = {}
    ts_scores = {}
    raw_scores = {}
    
    # ['note_embedding', 'note', 'asset', 'point', 'recording_chunks']

    # Calculate the cosine similarity between the query embedding and each chunk's embedding
    for doc_id, doc in vector_store.items():
        ####################################################################################
        # Compute similarity for annotations
        # Compute similarity for point properties
        # Compute similarity for asset properties x
        # Compute similarity for recording properties
        #
        # Ask LLM to analyse the returned data to provide fault diagnosis
        # Test how well it analyses spectra with good technical prompts
        # Can send in naked spectra, but also add more data e.g. point, prior notes, etc. , which LLM uses to enhance the diagnoses
        #
        # Retrieve closest data given the input data and distance metric evaluations that are provided, with a "standard" approach coded
        ####################################################################################

        asset = doc['asset']
        point = doc['point']
        note = doc['note']
        note_embedding = doc['note_embedding']
        
        if any(input_asset):
            # compute_path_distance
            path_score, asset_bleu = compute_path_short(input_asset['Path'], asset['Path'])
            asset_scores[doc_id] = path_score
            
        if any(input_point):
            # compute_bleu_distance
            point_score, detection_name_sim = compute_bleu_short(input_point, point)
            point_scores[doc_id] = point_score
            
        if any(input_note):
            # compute embedding_dot_product                       
            note_score = compute_score(query_note_embedding, norm_note_query, np.array(note_embedding[0]))
            input_note_time = parse(input_note['noteDateUTC'])
            note_time = parse(note['noteDateUTC'])
            note_time_diff = input_note_time - note_time
            note_date_sim = timedelta(100)-note_time_diff #1/(np.square(note_time_diff)+epsilon)
            note_scores[doc_id] = note_score

            
        if 'recording_chunks' in doc.keys():
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                                
                embedding_array = np.array(recording_chunk['signal_embedding'][0:3200])
                # norm_chunk = np.linalg.norm(embedding_array)
                
                # recording_span = [200:3200]
                spectrum_scores_list = []
                raw_spectrum_scores_list = []
                trends_scores_list, bias_scores_list, recording_tds_list, td_tds_list, speed_overlaps_list = [], [], [], [], []
                concat_scores_list = []
                ts_scores_list = []
                
                for query_chunk, query_spectrum_embedding, norm_spectrum_query, concat_query_embedding, concat_query_norm in zip(query_chunks, query_spectrum_embeddings, norm_spectrum_queries, concat_query_embeddings, concat_query_norms):
                    metadata = query_chunk['metadata']
                    spectra = metadata['spectra']
                    
                    spectrum_score = compute_score(query_spectrum_embedding[0:3200], norm_spectrum_query, embedding_array[0:3200])
                    spectrum_scores_list.append(spectrum_score)
                    raw_spectrum_score = compute_score(query_spectra[0:3200], raw_norm_spectrum_query, spectra[0:3200])
                    raw_spectrum_scores_list.append(raw_spectrum_score)
                    trends_score, bias_score, recording_td, td_td, speed_overlap, ts_score = compute_recording_scores(query_chunk, recording_chunk, conditions = False)
                    trends_scores_list.append(trends_score)
                    bias_scores_list.append(bias_score)
                    recording_tds_list.append(recording_td)
                    td_tds_list.append(td_td)
                    speed_overlaps_list.append(speed_overlap)
                    ts_scores_list.append(ts_score)
                    
                    # compute_score with both spectrum embedding and note embedding jointly
                    if any(input_note):
                        concat_embedding = np.concatenate((embedding_array[0:3200], np.array(note_embedding[0])))
                        concat_score = compute_score(concat_query_embedding, concat_query_norm, concat_embedding)
                        concat_scores_list.append(concat_score)
                    
                scores[(doc_id, chunk_id)] = np.average(spectrum_scores_list)
                trends_scores[(doc_id, chunk_id)] = np.average(trends_scores_list)
                bias_scores[(doc_id, chunk_id)] = np.average(bias_scores_list)
                recording_tds[(doc_id, chunk_id)] = np.average(recording_tds_list)
                td_tds[(doc_id, chunk_id)] = np.average(td_tds_list)
                speed_overlaps[(doc_id, chunk_id)] = np.average(speed_overlaps_list)
                concat_scores[(doc_id, chunk_id)] = np.average(concat_scores_list)
                ts_scores[(doc_id, chunk_id)] = np.average(ts_scores_list)
                raw_scores[(doc_id, chunk_id)] = np.average(raw_spectrum_scores_list)
    
    # Sort scores and return the top_k results
    top_results = sort_scores(scores, top_k)
    top_results_trends = sort_scores(trends_scores, top_k)
    top_results_bias = sort_scores(bias_scores, top_k)
    top_results_recording_tds = sort_scores(recording_tds, top_k)
    top_results_tds = sort_scores(td_tds, top_k)
    top_results_speeds = sort_scores(speed_overlaps, top_k)
    top_results_ts = sort_scores(ts_scores, top_k)
    top_results_raw = sort_scores(raw_scores, top_k)
    
    
    avg_scores = summarise_scores(scores, trends_scores, bias_scores, recording_tds, td_tds, speed_overlaps, weights = weights)
    top_results_avg = sort_scores(avg_scores, top_k)

    top_results_concat = sort_scores(concat_scores, top_k)
    
    top_results_assets = sort_asset_level_scores(asset_scores, top_k)
    top_results_points = sort_asset_level_scores(point_scores, top_k)
    top_results_notes = sort_asset_level_scores(note_scores, top_k)

    avg_scores_hierarchy = summarise_hierarchy_scores(asset_scores, point_scores, note_scores, asset_weights = asset_weights)
    top_results_hierarchy = sort_asset_level_scores(avg_scores_hierarchy, top_k)
    #return top_results, top_results_trends, top_results_bias, top_results_recording_tds, top_results_tds, top_results_speeds, top_results_avg, top_results_assets, top_results_points, top_results_notes, top_results_hierarchy, top_results_concat
    return top_results, top_results_trends, top_results_bias, top_results_recording_tds, top_results_tds, top_results_speeds, top_results_avg, top_results_assets, top_results_points, top_results_notes, top_results_hierarchy, top_results_concat, top_results_ts, top_results_raw


def sort_scores(scores, top_k):
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_results = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]
    return top_results

def sort_asset_level_scores(scores, top_k):
    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    top_results = [(doc_id, score) for ((doc_id), score) in sorted_scores]
    return top_results 

#def filter_vector_store(vector_store, asset_conditions, point_conditions, note_conditions, recording_conditions):
def filter_vector_store(vector_store, asset_conditions = [], point_conditions = [], note_conditions = [], recording_conditions = [], inverse = True):    
    # can we use existing filter functions for these?
    # We want to be able to call both and an or filters?
    # or just AND filters?
    # why would you ever want external OR? point with property X OR assets with property Y seems really edge case.
    # what about internal AND vs OR? It's fair to want both cable AND  
    # conditions = [{key_1: condition_1} etc...]
    # each doc in vector store has one point and one note, so don't have to bother with reconstructions, just inclusions, on the hierarchy level.
    # On the recording level, filtering should be made available for each recording, but this can also be modelled with just appending the recording or not.
    # Let input be any source key, condition key, condition value, and internal OR/AND
    # How to know the data source? Add source input
    # How to deal with recording keys? Still same input structure, just have to loop over recording chunks first
    # Make a different function that filters recordings? Let's start with hierarchy
    new_doc = {}
    for doc_id, doc in tqdm(vector_store.items()):
        conditions_fulfilled = False # If True at the end, we will add this part of the doc to the 
        asset_found, point_found, note_found = True, True, True
        asset = copy.deepcopy(doc['asset'])
        new_doc_slice = {}
        if any(asset_conditions):
            # each dict has to be fulfilled according to its criteria
            # for instance, the input [{"OR": ["TC1", "TC2"]}, {"AND": "TG1"}] requires both either TC1 or TC2 to be in the path, AND TG1 to be 
            # in the path. Most normal queries can be handled this way, except wanting an OR TG1, but that is less likely to be super relevant anyways.
            asset_found = look_in_asset(asset, asset_conditions)
            
        point = copy.deepcopy(doc['point'])
        if any(point_conditions):
            point_found = look_in_point(point, point_conditions)
            if inverse:
                point_found = not(point_found)
                
        note = copy.deepcopy(doc['note'])
        if any(note_conditions):
            note_found = look_in_note(note, note_conditions)
            
        recording = copy.deepcopy(doc['recording_chunks'])
        if any(recording_conditions):
            new_recording_chunk = look_in_recording(recording, recording_conditions)
        else:
            new_recording_chunk = recording
            
        if asset_found and point_found and note_found:
            new_doc[doc_id] = {'asset': asset, 'point': point, 'note': note, 'recording_chunks': new_recording_chunk}
    return new_doc

def look_in_recording(recording, recording_conditions):
    new_recording_chunk = {}
    for chunk_id, recording_chunk in recording.items():
        metadata = recording_chunk['metadata']
        recording_td = metadata['timedelta']
        if recording_td > recording_conditions[0] and recording_td < recording_conditions[1]:
            new_recording_chunk[chunk_id]= recording_chunk
    return new_recording_chunk
            
def look_in_asset(asset, asset_conditions):
    asset_path = asset['Path']
    asset_path_str = ' '.join(asset_path.split('\\'))
    
    # each dict has to be fulfilled according to its criteria
    # for instance, the input [{"OR": ["TC1", "TC2"]}, {"AND": "TG1"}] requires both either TC1 or TC2 to be in the path, AND TG1 to be 
    # in the path. Most normal queries can be handled this way, except wanting an OR TG1, but that is less likely to be super relevant anyways.
    conditions_true_counter = 0
    for asset_condition in asset_conditions:
        asset_found = False
        asset_found_counter = 0
        condition_logics, conditions = list(asset_condition.items())[0]
        asset_key = condition_logics[0]
        asset_logic = condition_logics[1]
        if asset_key not in ['Path']:
            return """Wrong Key! Please use the 'Path' key"""
        for condition in conditions:
            if condition.lower() in asset_path_str.lower():
                if asset_logic == 'OR':
                    asset_found = True
                    conditions_true_counter += 1
                    break
                elif asset_logic == 'AND':
                    asset_found_counter += 1
                else:
                    return """Wrong Logic Key! Please use either OR or AND"""
        if asset_logic == 'AND' and asset_found_counter == len(conditions):
            asset_found = True
            conditions_true_counter += 1
    if conditions_true_counter == len(asset_conditions):
        asset_found = True
    else:
        asset_found = False
    return asset_found
    
def look_in_point(point, point_conditions):
    # look for Name, DetectionName
    
    conditions_true_counter = 0 
    for point_condition in point_conditions:
        point_found = False
        point_found_counter = 0
        condition_logics, conditions = list(point_condition.items())[0]
        point_key = condition_logics[0]
        condition_logic = condition_logics[1]
        if point_key not in ['Name', 'DetectionName']:
            return """Wrong Key! Please use either the 'Name' or the 'DetectionName' key"""
        target_data = point[condition_logics[0]]
        
        for condition in conditions:
            if condition.lower() in target_data.lower():
                if condition_logics[1] == 'OR':
                    point_found = True
                    conditions_true_counter += 1
                    break
                elif condition_logics[1] == 'AND':
                    point_found_counter += 1
                else:
                    return """Wrong Logic Key! Please use either OR or AND"""
        if condition_logic == 'AND' and point_found_counter == len(conditions):
            point_found = True
            conditions_true_counter += 1
    if conditions_true_counter == len(point_conditions):
        point_found = True
    else:
        point_found = False
    return point_found

def look_in_note(note, note_conditions):
    # look for nodeName, noteDateUTC, title?, noteComment

    conditions_true_counter = 0 
    for note_condition in note_conditions:
        note_found = False
        note_found_counter = 0
        condition_logics, conditions = list(note_condition.items())[0]
        note_key = condition_logics[0]
        condition_logic = condition_logics[1]
        if note_key not in ['noteDateUTC', 'title', 'noteComment']:
            print("""Wrong Key! Please use either the 'noteDateUTC', 'title', or 'noteComment' key""")
            return """Wrong Key! Please use either the 'noteDateUTC', 'title', or 'noteComment' key"""
        target_data = note[note_key]
       
        for condition in conditions:
            if note_key == 'noteDateUTC':
                note_date = parse(target_data)
                note_in_time_span = note_date > condition[0] and note_date < condition[1] #condition has to be a time span list
                if note_in_time_span:
                    if condition_logic == 'OR':
                        note_found = True
                        conditions_true_counter += 1
                        break
                    elif condition_logic == 'AND':
                        note_found_counter += 1
                    else:
                        print("""Wrong Logic Key! Please use either OR or AND""")
                        return """Wrong Logic Key! Please use either OR or AND"""
            else:
                if condition.lower() in target_data.lower():
                    if condition_logic == 'OR':
                        note_found = True
                        conditions_true_counter += 1
                        break
                    elif condition_logic == 'AND':
                        note_found_counter += 1
                    else:

                        print("""Wrong Logic Key! Please use either OR or AND""")
                        return """Wrong Logic Key! Please use either OR or AND"""
        if condition_logic == 'AND' and note_found_counter == len(conditions):
            note_found = True
            conditions_true_counter += 1
    if conditions_true_counter == len(note_conditions):
        note_found = True
        # print(target_data)
        # print(condition)
        # print('')
    else:
        note_found = False
    return note_found

def summarise_scores(scores, trends_scores, bias_scores, recording_tds, td_tds, speed_overlaps, weights = [6, 3, 1, 0.5, 0.5, 0.0005]):
    '''
    Computes an average over multiple scoring metrics for SIGRAG.
    Weights affect, in order:
        spectrum embedding similarity
        trend similarity
        bias variance similarity
        recording date similarity
        recording td from note similarity
        speed similarity
    
    '''
    avg_scores = {}
    for (score_id, score), (trends_id, trends_score), (bias_id, bias_score), (recording_id, recording_td), (td_id, td_td), (speed_id, speed_overlap) in zip(scores.items(), trends_scores.items(), bias_scores.items(), recording_tds.items(), td_tds.items(), speed_overlaps.items()):
        avg_score = weights[0]*score+weights[1]*trends_score+weights[2]*bias_score+weights[3]*recording_td+weights[4]*td_td+weights[5]*speed_overlap
        avg_scores[score_id] = avg_score
    return avg_scores

def summarise_hierarchy_scores(assets_scores, points_scores, notes_scores, asset_weights = [1, 1, 1]):
    '''
    Computes an average over multiple scoring metrics for SIGRAG.
    Weights affect, in order:
        asset
        point
        note
    
    '''
    weights = asset_weights
    avg_scores = {}
    for (assets_score_id, assets_score), (points_score_id, points_score), (notes_score_id, notes_score) in zip(assets_scores.items(), points_scores.items(), notes_scores.items()):
        avg_score = weights[0]*assets_score+weights[1]*points_score+weights[2]*notes_score
        avg_scores[assets_score_id] = avg_score
    return avg_scores

def infer_point_information(documents, top_results):
    # point_labels = ['FU', 'FÖ', 'MO', 'TC', 'VX']
    # point_votes = [0, 0, 0, 0]
    points_dict = {'FU': 0, 'FÖ': 0, 'MO': 0, 'TC': 0, 'VX': 0}
    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []
    for top_result in top_results:
        top_doc_id = top_result[0]
        top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[1]
        top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        top_chunk_scores.append(top_chunk_score)
        
    for doc_id, doc in documents.items():
        if doc_id in top_doc_ids:
            asset = doc['asset']
            point = doc['point']
            note = doc['note']
            point_name = point['Name'][0:2]
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                if chunk_id in top_chunk_ids:
                    points_dict[point_name]+=1
            
    return points_dict

def infer_information(documents, top_results):
    '''
    Takes the vector store and the list of nearest SIGRAG elements and makes a dict of guesses based on element properties
    '''
    note_labels = [["bytt"], ["kabel", "giv", "sens"], ["bpfo", "bpfi", "bsf", "lager", "lagret"], ["glapp", "haveri", "obalans"]]
    # point_labels = ['FU', 'FÖ', 'MO', 'TC']
    # point_votes = [0, 0, 0, 0]
    # points_dict = {'FU': 0, 'FÖ': 0, 'MO': 0, 'TC': 0}
    points_dict = {}
    notes_label_names = ['replaced', 'cable_sensor_faults', 'bearing_faults', 'critical_faults', 'misc']
    # notes_dict = {'replaced': 0, 'cable_sensor_faults': 0, 'bearing_faults': 0, 'critical_faults': 0, 'misc': 0}
    notes_dict = {}
    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []

    for top_result in top_results:
        top_doc_id = top_result[0]
        # top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[1]
        # top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        # top_chunk_scores.append(top_chunk_score)
        
   
        for doc_id, doc in documents.items():
            if doc_id == top_doc_id:
                asset = doc['asset']
                point = doc['point']
                note = doc['note']
                notecomment = note['noteComment'].lower()
                # breakflag = False
                # for note_label_nr, note_label in enumerate(note_labels):
                #     for note_lab in note_label:
                #         if note_lab in notecomment:
                #             notes_dict[notes_label_names[note_label_nr]]+=1
                #             breakflag = True
                #     if breakflag:
                #         break
                # if not breakflag:
                #     notes_dict['misc']+=1
                
                note_label_value = make_note_label(notecomment, note_labels, notes_label_names)
    
                pointname = point['Name']
                point_label_value = make_point_label(pointname)
                # if point_label_value in points_dict:
                #     points_dict[point_label_value]+=1
                # else:
                #     points_dict[point_label_value]=1
                for chunk_id, recording_chunk in doc['recording_chunks'].items():
                    if chunk_id == top_chunk_id:
                        if point_label_value in points_dict:
                            points_dict[point_label_value]+=top_chunk_score
                        else:
                            points_dict[point_label_value]=top_chunk_score
                        if note_label_value in notes_dict:
                            notes_dict[note_label_value]+=top_chunk_score
                        else:
                            notes_dict[note_label_value]=top_chunk_score
                        # print(points_dict)
                        # print(notes_dict)
                # if 'MO' in pointname:
                #     point_name = 'MO'
                # elif 'VXL' in pointname:
                #     point_name = 'VXL'
                # else:
                #     # print(pointname)
                #     point_name = point['Name'].split(' ')[0]
                
    return points_dict, notes_dict
                    

def plot_input_chunks(documents, doc_key, chunk_keys):
    input_chunks = []
    for doc_id, doc in documents.items():
        if doc_id == doc_key:
            input_asset = doc['asset']
            input_points = doc['point']
            input_note = doc['note']
            # print(input_asset['Path'])
            
            for recording_id, recording in doc['recording_chunks'].items():
                print(recording_id)
                for chunk_key in chunk_keys:
                    if recording_id == chunk_key:
                        input_chunks.append(recording)
    if any(input_chunks):
        for input_chunk in input_chunks:
            input_signal_embedding, input_metadata = input_chunk.items()
            # print(input_metadata)
            legend = ''
            if any(input_asset):
                legend+='Asset: '+input_asset['Path']+'\n'
        
            if any(input_points):
                legend+='Point: '+input_points['Name']+ ' at time: ' +str(input_metadata[1]['dates']) + ' with speed: ' +str(input_metadata[1]['speeds'])+'\n'
        
            if any(input_note):
                # compute embedding_dot_product
                note_time = parse(input_note['noteDateUTC'])
                note_content = input_note['noteComment']
                input_td = input_metadata[1]['timedelta']
                legend+='Note '+str(input_td) + ' away: ' + note_content
    
            input_spectra = input_metadata[1]['spectra']
            plt.plot(input_spectra, color = 'g')
            plt.legend([legend])
            plt.show()
        return "Plots plotted OK"
    else:
        return "No recordings were identified"

def plot_chunk(vector_store, top_results, input_chunk, input_asset, input_points, input_note):
    input_signal_embedding, input_metadata = input_chunk.items()
    # print(input_metadata)
    legend = ''
    if any(input_asset):
        legend+='Asset: '+input_asset['Path']+'\n'

    if any(input_points):
        legend+='Point: '+input_points['Name']+ ' at time: ' +str(input_metadata[1]['dates']) + ' with speed: ' +str(input_metadata[1]['speeds'])+'\n'

    if any(input_note):
        # compute embedding_dot_product
        input_note_time = parse(input_note['noteDateUTC'])
        note_time = parse(input_note['noteDateUTC'])
        note_content = input_note['noteComment']
        input_td = input_metadata[1]['timedelta']
        legend+='Note '+str(input_td) + ' away: ' + note_content
    plt.plot(input_metadata[1]['spectra'])
    plt.legend([legend])
    plt.show()
    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []
    for top_result in top_results:
        top_doc_id = top_result[0]
        top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[1]
        top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        top_chunk_scores.append(top_chunk_score)
        
    for doc_id, doc in vector_store.items():
        if doc_id in top_doc_ids:
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                if chunk_id in top_chunk_ids:    
                    embedding_array = np.array(recording_chunk['signal_embedding'])
                    metadata = recording_chunk['metadata']
                    recording_td = metadata['timedelta']
                    asset = doc['asset']
                    point = doc['point']
                    note = doc['note']
                    plt.plot(embedding_array)
                    legend = 'Asset: '+asset['Path']+'\n'+'Point: '+point['Name']+ ' at time: ' +str(metadata['dates'])+ ' with speed: ' +str(metadata['speeds'])+'\n'+'Note '+str(recording_td) +' away: ' + note['noteComment']
                    plt.legend([legend])
                    plt.show()
            # note_embedding = doc['note_embedding']


def get_hierarchy(vector_store, top_results):

    assets = []
    points = []
    notes = []
    tds = []
    for top_result in top_results:
        top_doc_id = top_result[0]
        top_chunk_id = top_result[1]
        top_chunk_score = top_result[2]
        
        for doc_id, doc in vector_store.items():
            if doc_id == top_doc_id:
                for chunk_id, recording_chunk in doc['recording_chunks'].items():
                    if chunk_id == top_chunk_id:    
                        # embedding_array = np.array(recording_chunk['signal_embedding'])
                        metadata = recording_chunk['metadata']
                        
                        recording_td = metadata['timedelta'].days
                        tds.append({'time delta': recording_td})
                        # spectra = metadata['spectra']
                        assets.append({'asset path': ' '.join(doc['asset']['Path'].split('\\')[2:])})
                        points.append({'point name':doc['point']['Name']})
                        notes.append({'note date': doc['note']['noteDateUTC'], 'note content': doc['note']['noteComment']})
    return assets, points, notes, tds

def get_recording_from_chunk_key(vector_store, chunk_key):
    for doc_id, doc in vector_store.items():
        for chunk_id, recording_chunk in doc['recording_chunks'].items():
            if chunk_id == chunk_key:    
                # embedding_array = np.array(recording_chunk['signal_embedding'])
                metadata = recording_chunk['metadata']
                recording_td = metadata['timedelta']
                spectra = metadata['spectra']
                levels = metadata['levels']
                biases = metadata['bias']
                speeds = metadata['speeds']
    return spectra, levels, biases, speeds

def get_keys_from_chunk_keys(vector_store, chunk_key):
    chunk_keys = []
    for doc_id, doc in vector_store.items():
        chunk_key_buffer = []
        chunkflag = False
        for chunk_id, recording_chunk in doc['recording_chunks'].items():
            if chunk_id == chunk_key:
                chunk_doc_id = doc_id
                chunkflag = True
            else:
                chunk_key_buffer.append(chunk_id)
        if chunkflag:
            chunk_keys = chunk_key_buffer
    return chunk_doc_id, chunk_keys

def get_chunks_from_chunk_keys(vector_store, chunk_keys):
    chunks = {}
    for doc_id, doc in vector_store.items():
        chunkflag = False
        for chunk_id, recording_chunk in doc['recording_chunks'].items():
            if chunk_id in chunk_keys:
                chunks[chunk_id]=recording_chunk
    return chunks

def get_hierarchy_from_chunk_key(vector_store, chunk_key):
    for doc_id, doc in vector_store.items():
        for chunk_id, recording_chunk in doc['recording_chunks'].items():
            if chunk_id == chunk_key:
                assets = doc['asset']
                points = doc['point']
                notes = doc['note']
    return assets, points, notes

def plot_chunk_keys(vector_store, chunk_keys, savefig = True):
    return_embeddings = []
    for doc_id, doc in vector_store.items():
        for chunk_id, recording_chunk in doc['recording_chunks'].items():
            for chunk_key in chunk_keys:
                if chunk_id == chunk_key:    
                    embedding_array = np.array(recording_chunk['signal_embedding'])
                    # spectrum_embedding = np.zeros(3200)
                    # spectrum_embedding[0:len(embedding_array[0:3200])] = np.array(embedding_array[0:3200]) #np.array(spectrum_encoder(query_spectrum))
                    # spectrum_embedding = spectrum_embedding-np.average(spectrum_embedding)
                    # spectrum_embedding = np.where(spectrum_embedding<0, 0, spectrum_embedding)
                    # embedding_array = spectrum_embedding
                    
                    
                    metadata = recording_chunk['metadata']
                    recording_td = metadata['timedelta']
                    spectra = metadata['spectra']
                    levels = metadata['levels']
                    biases = metadata['bias']
                    speeds = metadata['speeds']
                    asset = doc['asset']
                    point = doc['point']
                    note = doc['note']
                    # fig, axs = plt.subplots(2)
                    # ax1 = axs[0]
                    fig, ax1 = plt.subplots(1)
                    
                    # ax1.plot(embedding_array)
                    ax1.plot(spectra)
                    
                    legend = 'Asset: '+asset['Path']+'\n'+'Point: '+point['Name']+ ' at time: ' +str(metadata['dates'])+ ' with speed: ' +str(metadata['speeds'])+'\n'+'Note '+str(recording_td) +' away: ' + note['noteComment']
                    ax1.legend([legend])
                    ax1.set_xlabel('order index')
                    ax1.set_ylabel('envelope value')
                    # plt.show()
                    
                    # color = 'tab:red'
                    # ax2 = axs[1]
                    # ax2.plot(levels, color=color)
                    # ax2.tick_params(axis='y', labelcolor=color)
                    # ax2.set_ylabel('level', color=color)
                    # ax2.set_xlabel('time')
                    # ax2_1 = ax2.twinx()
                    
                    # color = 'tab:blue'
                    # ax2_1.plot(biases, color=color)
                    # ax2_1.tick_params(axis='y', labelcolor=color)
                    # ax2_1.set_ylabel('bias', color=color)
                    fig.tight_layout()
                    if savefig:
                        fig.savefig("input_chunk"+str(doc_id)+str(chunk_id)+".pdf", bbox_inches='tight')
                    plt.show()
                    return_embeddings.append(spectra)
    return return_embeddings

def get_spectra(vector_store, top_results, top_k = 3):
    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []
    levels = []
    biases = []
    speeds = []
    recording_tds = []
    for top_result in top_results[0:top_k]:
        top_doc_id = top_result[0]
        top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[1]
        top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        top_chunk_scores.append(top_chunk_score)
        
    spectra = []
    for doc_id, doc in vector_store.items():
        if doc_id in top_doc_ids:
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                if chunk_id in top_chunk_ids:    
                    # embedding_array = np.array(recording_chunk['signal_embedding'])
                    metadata = recording_chunk['metadata']
                    recording_tds.append(metadata['timedelta'])
                    spectra.append(metadata['spectra'])
                    levels.append(metadata['levels'])
                    biases.append(metadata['bias'])
                    speeds.append(metadata['speeds'])
    return spectra, levels, biases, speeds
            

def get_recording(vector_store, top_results, top_k = 3):
    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []
    levels = []
    biases = []
    speeds = []
    recording_tds = []
    for top_result in top_results[0:top_k]:
        top_doc_id = top_result[0]
        top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[1]
        top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        top_chunk_scores.append(top_chunk_score)
        
    spectra = []
    for doc_id, doc in vector_store.items():
        if doc_id in top_doc_ids:
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                if chunk_id in top_chunk_ids:    
                    # embedding_array = np.array(recording_chunk['signal_embedding'])
                    metadata = recording_chunk['metadata']
                    recording_tds.append(metadata['timedelta'])
                    spectra.append(metadata['spectra'])
                    levels.append(metadata['levels'])
                    biases.append(metadata['bias'])
                    speeds.append(metadata['speeds'])
    return spectra, levels, biases, speeds, recording_tds


def plot_chunks(vector_store, top_results, input_chunks = [], input_asset = [], input_points = [], input_note = []):
    for input_chunk in input_chunks:
        input_signal_embedding, input_metadata = input_chunk.items()
        # print(input_metadata)
        legend = ''
        if any(input_asset):
            legend+='Asset: '+input_asset['Path']+'\n'
    
        if any(input_points):
            legend+='Point: '+input_points['Name']+ ' at time: ' +str(input_metadata[1]['dates']) + ' with speed: ' +str(input_metadata[1]['speeds'])+'\n'
    
        if any(input_note):
            # compute embedding_dot_product
            input_note_time = parse(input_note['noteDateUTC'])
            note_time = parse(input_note['noteDateUTC'])
            note_content = input_note['noteComment']
            input_td = input_metadata[1]['timedelta']
            legend+='Note '+str(input_td) + ' away: ' + note_content
        input_spectra = input_metadata[1]['spectra']
        plt.plot(input_spectra, color = 'g')
        plt.legend([legend])
        plt.show()

    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []
    for top_result in top_results:
        top_doc_id = top_result[0]
        top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[1]
        top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        top_chunk_scores.append(top_chunk_score)
        
    for doc_id, doc in vector_store.items():
        if doc_id in top_doc_ids:
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                if chunk_id in top_chunk_ids:    
                    # embedding_array = np.array(recording_chunk['signal_embedding'])
                    metadata = recording_chunk['metadata']
                    recording_td = metadata['timedelta']
                    spectra = metadata['spectra']
                    asset = doc['asset']
                    point = doc['point']
                    note = doc['note']
                    plt.plot(spectra)
                    legend = 'Asset: '+asset['Path']+'\n'+'Point: '+point['Name']+ ' at time: ' +str(metadata['dates'])+ ' with speed: ' +str(metadata['speeds'])+'\n'+'Note '+str(recording_td) +' away: ' + note['noteComment']
                    plt.legend([legend])
                    plt.show()
            # note_embedding = doc['note_embedding']

def new_plot_target_chunks(vector_store, top_results, savefig = True):
    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []
    return_embeddings = []
    for top_result in top_results:
        
        top_doc_id = top_result[0][0]
        top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[0][1]
        top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        top_chunk_scores.append(top_chunk_score)
        
    for doc_id, doc in vector_store.items():
        if doc_id in top_doc_ids:
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                if chunk_id in top_chunk_ids:    
                    embedding_array = np.array(recording_chunk['signal_embedding'])
                    
                    # spectrum_embedding = np.zeros(3200)
                    # spectrum_embedding[0:len(embedding_array[0:3200])] = np.array(embedding_array[0:3200]) #np.array(spectrum_encoder(query_spectrum))
                    # spectrum_embedding = spectrum_embedding-np.average(spectrum_embedding)
                    # spectrum_embedding = np.where(spectrum_embedding<0, 0, spectrum_embedding)
                    # embedding_array = spectrum_embedding
                    
                    # metadata = recording_chunk['metadata']
                    # recording_td = metadata['timedelta']
                    # spectra = metadata['spectra']
                    # asset = doc['asset']
                    # point = doc['point']
                    # note = doc['note']
                    # plt.plot(spectra)
                    # legend = 'Asset: '+asset['Path']+'\n'+'Point: '+point['Name']+ ' at time: ' +str(metadata['dates'])+ ' with speed: ' +str(metadata['speeds'])+'\n'+'Note '+str(recording_td) +' away: ' + note['noteComment']
                    # plt.legend([legend])
                    # plt.show()
                    
                    
                    metadata = recording_chunk['metadata']
                    recording_td = metadata['timedelta']
                    spectra = metadata['spectra']
                    levels = metadata['levels']
                    biases = metadata['bias']
                    speeds = metadata['speeds']
                    asset = doc['asset']
                    point = doc['point']
                    note = doc['note']
                    # fig, axs = plt.subplots(2)
                    # ax1 = axs[0]
                    fig, ax1 = plt.subplots(1)
                    
                    
                    # ax1.plot(embedding_array)
                    ax1.plot(spectra)
                    
                    
                    legend = 'Asset: '+asset['Path']+'\n'+'Point: '+point['Name']+ ' at time: ' +str(metadata['dates'])+ ' with speed: ' +str(metadata['speeds'])+'\n'+'Note '+str(recording_td) +' away: ' + note['noteComment']
                    ax1.legend([legend])
                    ax1.set_xlabel('order index')
                    ax1.set_ylabel('envelope value')
                    # plt.show()
                    
                    # color = 'tab:red'
                    # ax2 = axs[1]
                    # ax2.plot(levels, color=color)
                    # ax2.tick_params(axis='y', labelcolor=color)
                    # ax2.set_ylabel('level', color=color)
                    # ax2.set_xlabel('time')
                    # ax2_1 = ax2.twinx()
                    
                    # color = 'tab:blue'
                    # ax2_1.plot(biases, color=color)
                    # ax2_1.tick_params(axis='y', labelcolor=color)
                    # ax2_1.set_ylabel('bias', color=color)
                    fig.tight_layout()
                    if savefig:
                        fig.savefig("output_chunk"+str(doc_id)+str(chunk_id)+".pdf", bbox_inches='tight')
                    plt.show()
                    
                    return_embeddings.append(spectra)
    return return_embeddings

def plot_target_chunks(vector_store, top_results):
    top_doc_ids = []
    top_chunk_ids = []
    top_chunk_scores = []
    for top_result in top_results:
        top_doc_id = top_result[0]
        top_doc_ids.append(top_doc_id)
        top_chunk_id = top_result[1]
        top_chunk_ids.append(top_chunk_id)
        top_chunk_score = top_result[2]
        top_chunk_scores.append(top_chunk_score)
        
    for doc_id, doc in vector_store.items():
        if doc_id in top_doc_ids:
            for chunk_id, recording_chunk in doc['recording_chunks'].items():
                if chunk_id in top_chunk_ids:    
                    # embedding_array = np.array(recording_chunk['signal_embedding'])
                    metadata = recording_chunk['metadata']
                    recording_td = metadata['timedelta']
                    spectra = metadata['spectra']
                    asset = doc['asset']
                    point = doc['point']
                    note = doc['note']
                    plt.plot(spectra)
                    legend = 'Asset: '+asset['Path']+'\n'+'Point: '+point['Name']+ ' at time: ' +str(metadata['dates'])+ ' with speed: ' +str(metadata['speeds'])+'\n'+'Note '+str(recording_td) +' away: ' + note['noteComment']
                    plt.legend([legend])
                    plt.show()

def make_predictions(documents, use_same_point = False, top_k = 10):
    from tqdm import tqdm
    point_labels = []
    note_labels = []
    emb_guesses_points, emb_guesses_notes = [], []
    emb_preds_points, emb_preds_notes = [], []
    ts_guesses_points, ts_guesses_notes = [], []
    ts_preds_points, ts_preds_notes = [], []
    sp_guesses_points, sp_guesses_notes = [], []
    sp_preds_points, sp_preds_notes = [], []
    emb_preds_assets = []
    ts_preds_assets = []
    sp_preds_assets = []
    asset_point_labels = []
    emb_preds_assets_notes = []
    ts_preds_assets_notes = []
    sp_preds_assets_notes = []
    asset_notes_labels = []

    for doc_id, doc in tqdm(documents.items()):
        target_asset = doc['asset']
        target_points = doc['point']
        pointname = target_points['Name']
        if 'MO' in pointname:
            point_type = 'MO'
        elif 'VXL' in pointname:
            point_type = 'VXL'
        else:
            # print(pointname)
            point_type = pointname.split(' ')[0]
        asset_point_labels.append(point_type)
        
        target_note = doc['note']
        target_notecontent = target_note['noteComment']
        note_label_contents = [["bytt"], ["kabel", "giv", "sens"], ["bpfo", "bpfi", "bsf", "lager", "lagret"], ["glapp", "haveri", "obalans"]]
        notes_label_names = ['replaced', 'cable_sensor_faults', 'bearing_faults', 'critical_faults', 'misc']
        
        breakflag = False
        # print(notecomment)
        for note_label_nr, note_label in enumerate(note_label_contents):
            # print(note_label)
            for note_lab in note_label:
                # print(note_lab)
                if note_lab.lower() in target_notecontent.lower():
                    note_label_value = notes_label_names[note_label_nr]
                    breakflag = True
            if breakflag:
                break
        if not breakflag:
            note_label_value = 'misc'
        asset_notes_labels.append(note_label_value)
        
        if use_same_point:
            filtered_doc = documents
        else:
            point_conditions = [{('Name', 'OR'): [target_points['Name']]}]
            filtered_doc = filter_vector_store(documents, asset_conditions = [], point_conditions = point_conditions, note_conditions = [])
        query_chunks = []
        for recording_id, recording in doc['recording_chunks'].items():
            query_chunks.append(recording)
        emb_scores, time_series_scores, raw_spectrum_scores = compute_recordings_embedding_distance(filtered_doc, query_chunks, top_k = top_k) 
        emb_guess_points_buffer = []
        ts_guess_points_buffer = []
        raw_guess_points_buffer = []
        emb_guess_notes_buffer = []
        ts_guess_notes_buffer = []
        raw_guess_notes_buffer = []
        
        for emb_score, time_series_score, raw_spectrum_score in zip(emb_scores, time_series_scores, raw_spectrum_scores):
            point_labels.append(point_type)
            note_labels.append(note_label_value)
            
            sorted_emb_scores = sorted(emb_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
            top_results_emb = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_emb_scores]
            
            
            # store and return top_results_emb etc. instead, and make another func that goes from that to guesses
            
            emb_results_guesses_points, emb_results_guesses_notes = infer_information(filtered_doc, top_results_emb)
            emb_guesses_points.append(emb_results_guesses_points)
            emb_guess_points = max(emb_results_guesses_points, key=emb_results_guesses_points.get)
            emb_preds_points.append(emb_guess_points)
            emb_guess_points_buffer.append(emb_guess_points)
            emb_guesses_notes.append(emb_results_guesses_notes)
            emb_guess_notes = max(emb_results_guesses_notes, key=emb_results_guesses_notes.get)
            emb_preds_notes.append(emb_guess_notes)
            emb_guess_notes_buffer.append(emb_guess_notes)
    
            sorted_ts_scores = sorted(time_series_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
            top_results_ts = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_ts_scores]
            ts_results_guesses_points, ts_results_guesses_notes = infer_information(filtered_doc, top_results_ts)
            ts_guesses_points.append(ts_results_guesses_points)
            ts_guess_points = max(ts_results_guesses_points, key=ts_results_guesses_points.get)
            ts_guess_points_buffer.append(ts_guess_points)
            ts_preds_points.append(ts_guess_points)
            ts_guesses_notes.append(ts_results_guesses_notes)
            ts_guess_notes = max(ts_results_guesses_notes, key=ts_results_guesses_notes.get)
            ts_preds_notes.append(ts_guess_notes)
            ts_guess_notes_buffer.append(ts_guess_notes)
    
            sorted_raw_scores = sorted(raw_spectrum_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
            top_results_raw = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_raw_scores]
            # print("top_results_raw")
            # print(top_results_raw)
            raw_results_guesses_points, raw_results_guesses_notes = infer_information(filtered_doc, top_results_raw)
            sp_guesses_points.append(raw_results_guesses_points)

            raw_guess_points = max(raw_results_guesses_points, key=raw_results_guesses_points.get)
            raw_guess_points_buffer.append(raw_guess_points)
            sp_preds_points.append(raw_guess_points)
            sp_guesses_notes.append(raw_results_guesses_notes)
            raw_guess_notes = max(raw_results_guesses_notes, key=raw_results_guesses_notes.get)
            sp_preds_notes.append(raw_guess_notes)
            raw_guess_notes_buffer.append(raw_guess_notes)
            
        emb_preds_assets.append(max(emb_guess_points_buffer))
        ts_preds_assets.append(max(ts_guess_points_buffer))
        sp_preds_assets.append(max(raw_guess_points_buffer))
        emb_preds_assets_notes.append(max(emb_guess_notes_buffer))
        ts_preds_assets_notes.append(max(ts_guess_notes_buffer))
        sp_preds_assets_notes.append(max(raw_guess_notes_buffer))
        
    return point_labels, emb_preds_points, ts_preds_points, sp_preds_points, note_labels, emb_preds_notes, ts_preds_notes, sp_preds_notes, asset_point_labels, emb_preds_assets, ts_preds_assets, sp_preds_assets, asset_notes_labels, emb_preds_assets_notes, ts_preds_assets_notes, sp_preds_assets_notes

def compute_sim(documents, use_same_point = False, top_k = 10):
    '''
    Computes the similarity between all chunks in the data
    '''
    from tqdm import tqdm
    point_labels = []
    note_labels = []
    asset_point_labels = []
    asset_notes_labels = []
    emb_scores_list, time_series_scores_list, raw_spectrum_scores_list = [], [], []
    for doc_id, doc in tqdm(documents.items()):
        target_asset = doc['asset']
        target_points = doc['point']
        pointname = target_points['Name']
        point_type = make_point_label(pointname)
        asset_point_labels.append(point_type)
        
        target_note = doc['note']
        target_notecontent = target_note['noteComment']
        
        note_label_value = make_note_label(target_notecontent)
        asset_notes_labels.append(note_label_value)
        
        if use_same_point:
            filtered_doc = documents
        else:
            point_conditions = [{('Name', 'OR'): [target_points['Name']]}]
            filtered_doc = filter_vector_store(documents, asset_conditions = [], point_conditions = point_conditions, note_conditions = [])
        query_chunks = []
        for recording_id, recording in doc['recording_chunks'].items():
            query_chunks.append(recording)
        emb_scores, time_series_scores, raw_spectrum_scores = compute_recordings_embedding_distance(filtered_doc, query_chunks, top_k = top_k) 
        emb_scores_list.append(emb_scores)
        time_series_scores_list.append(time_series_scores)
        raw_spectrum_scores_list.append(raw_spectrum_scores)
        
    return asset_point_labels, asset_notes_labels, emb_scores_list, time_series_scores_list, raw_spectrum_scores_list

def make_point_label(pointname):
    '''
    if 'MO'.lower() in pointname.lower():
        point_type = 'MO'
    elif 'VXL'.lower() in pointname.lower():
        point_type = 'VXL'
    elif 'vals'.lower() in pointname.lower():
        point_type = 'vals'
    elif 'vira'.lower() in pointname.lower():
        point_type = 'vals'
    else:
        # print(pointname)
        point_type = pointname.split(' ')[0]
    '''
    exclusion_criteria = ["HV", "HE1", "HE2", "HE3", "HE4",
                     "AV", "AE1", "AE2", "AE3", "AE4",
                     "VV", "VE1", "VE2", "VE3", "VE4",
                     "FS", "DS"]

    pointname_reconstructed = ''
    for pointnam in pointname.split():
        if pointnam not in exclusion_criteria:
            pointname_reconstructed+=pointnam + ' '
    return pointname_reconstructed[0:-1]

def make_note_label(notecontent, note_label_contents = False, notes_label_names = False):
    if not note_label_contents:
        note_label_contents = [["bytt"], ["kabel", "giv", "sens"], ["bpfo", "bpfi", "bsf", "lager", "lagret"], ["glapp", "haveri", "obalans"]]
    if not notes_label_names:
        notes_label_names = ['replaced', 'cable_sensor_faults', 'bearing_faults', 'critical_faults', 'misc']

    breakflag = False
    # print(notecomment)
    for note_label_nr, note_label in enumerate(note_label_contents):
        # print(note_label)
        for note_lab in note_label:
            # print(note_lab)
            if note_lab.lower() in notecontent.lower():
                note_label_value = notes_label_names[note_label_nr]
                breakflag = True
        if breakflag:
            break
    if not breakflag:
        note_label_value = 'misc'
    return note_label_value

def compute_query_sim(documents, query_keys, use_same_point = False):
    '''
    A wrapper for compute_recordings_id_embedding_distance that also returns labels, that runs over the whole doc


    Parameters
    ----------
    documents : TYPE
        DESCRIPTION.
    query_keys : TYPE
        DESCRIPTION.
    use_same_point : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    asset_point_labels : TYPE
        DESCRIPTION.
    asset_notes_labels : TYPE
        DESCRIPTION.
    emb_scores : TYPE
        DESCRIPTION.

    '''
    
    from tqdm import tqdm
    point_labels = []
    note_labels = []
    asset_point_labels = []
    asset_notes_labels = []
    emb_scores_list = []
    for doc_id, doc in tqdm(documents.items()):
        target_asset = doc['asset']
        target_points = doc['point']
        pointname = target_points['Name']
        point_type = make_point_label(pointname)
        asset_point_labels.append(point_type)
        
        target_note = doc['note']
        target_notecontent = target_note['noteComment']
        
        note_label_value = make_note_label(target_notecontent)
        asset_notes_labels.append(note_label_value)

        emb_scores = compute_recordings_id_embedding_distance(documents, query_keys, exclude_same_point = use_same_point) 
        
    return asset_point_labels, asset_notes_labels, emb_scores

def make_predictions_sim(filtered_doc, asset_point_labels, asset_notes_labels, emb_scores_list, time_series_scores_list, raw_spectrum_scores_list, top_k = 10):
    from tqdm import tqdm
    point_labels = []
    note_labels = []
    emb_guesses_points, emb_guesses_notes = [], []
    emb_preds_points, emb_preds_notes = [], []
    ts_guesses_points, ts_guesses_notes = [], []
    ts_preds_points, ts_preds_notes = [], []
    sp_guesses_points, sp_guesses_notes = [], []
    sp_preds_points, sp_preds_notes = [], []
    emb_preds_assets = []
    ts_preds_assets = []
    sp_preds_assets = []
    # asset_point_labels = []
    emb_preds_assets_notes = []
    ts_preds_assets_notes = []
    sp_preds_assets_notes = []
    # asset_notes_labels = []
    point_labels = []
    note_labels = []
    for asset_point_label, asset_notes_label, emb_scores, time_series_scores, raw_spectrum_scores in tqdm(zip(asset_point_labels, asset_notes_labels, emb_scores_list, time_series_scores_list, raw_spectrum_scores_list)):
        
        emb_guess_points_buffer = []
        ts_guess_points_buffer = []
        raw_guess_points_buffer = []
        emb_guess_notes_buffer = []
        ts_guess_notes_buffer = []
        raw_guess_notes_buffer = []
        for emb_score, time_series_score, raw_spectrum_score in zip(emb_scores, time_series_scores, raw_spectrum_scores):
            point_labels.append(asset_point_label)
            note_labels.append(asset_notes_label)
            
            sorted_emb_scores = sorted(emb_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
            top_results_emb = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_emb_scores]
            
            
            # store and return top_results_emb etc. instead, and make another func that goes from that to guesses
            
            emb_results_guesses_points, emb_results_guesses_notes = infer_information(filtered_doc, top_results_emb)
            emb_guesses_points.append(emb_results_guesses_points)
            emb_guess_points = max(emb_results_guesses_points, key=emb_results_guesses_points.get)
            emb_preds_points.append(emb_guess_points)
            emb_guess_points_buffer.append(emb_guess_points)
            emb_guesses_notes.append(emb_results_guesses_notes)
            emb_guess_notes = max(emb_results_guesses_notes, key=emb_results_guesses_notes.get)
            emb_preds_notes.append(emb_guess_notes)
            emb_guess_notes_buffer.append(emb_guess_notes)
    
            sorted_ts_scores = sorted(time_series_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
            top_results_ts = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_ts_scores]
            ts_results_guesses_points, ts_results_guesses_notes = infer_information(filtered_doc, top_results_ts)
            ts_guesses_points.append(ts_results_guesses_points)
            ts_guess_points = max(ts_results_guesses_points, key=ts_results_guesses_points.get)
            ts_guess_points_buffer.append(ts_guess_points)
            ts_preds_points.append(ts_guess_points)
            ts_guesses_notes.append(ts_results_guesses_notes)
            ts_guess_notes = max(ts_results_guesses_notes, key=ts_results_guesses_notes.get)
            ts_preds_notes.append(ts_guess_notes)
            ts_guess_notes_buffer.append(ts_guess_notes)
    
            sorted_raw_scores = sorted(raw_spectrum_score.items(), key=lambda item: item[1], reverse=True)[:top_k]
            top_results_raw = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_raw_scores]
            # print("top_results_raw")
            # print(top_results_raw)
            raw_results_guesses_points, raw_results_guesses_notes = infer_information(filtered_doc, top_results_raw)
            sp_guesses_points.append(raw_results_guesses_points)

            raw_guess_points = max(raw_results_guesses_points, key=raw_results_guesses_points.get)
            raw_guess_points_buffer.append(raw_guess_points)
            sp_preds_points.append(raw_guess_points)
            sp_guesses_notes.append(raw_results_guesses_notes)
            raw_guess_notes = max(raw_results_guesses_notes, key=raw_results_guesses_notes.get)
            sp_preds_notes.append(raw_guess_notes)
            raw_guess_notes_buffer.append(raw_guess_notes)
            
        emb_preds_assets.append(max(emb_guess_points_buffer))
        ts_preds_assets.append(max(ts_guess_points_buffer))
        sp_preds_assets.append(max(raw_guess_points_buffer))
        emb_preds_assets_notes.append(max(emb_guess_notes_buffer))
        ts_preds_assets_notes.append(max(ts_guess_notes_buffer))
        sp_preds_assets_notes.append(max(raw_guess_notes_buffer))
        
    return point_labels, emb_preds_points, ts_preds_points, sp_preds_points, note_labels, emb_preds_notes, ts_preds_notes, sp_preds_notes, asset_point_labels, emb_preds_assets, ts_preds_assets, sp_preds_assets, asset_notes_labels, emb_preds_assets_notes, ts_preds_assets_notes, sp_preds_assets_notes, emb_guesses_points, emb_guesses_notes

def make_predictions_emb(filtered_doc, asset_point_labels, asset_notes_labels, emb_scores_chunk, top_k = 10):
    from tqdm import tqdm
    point_labels = []
    note_labels = []
    emb_guesses_points, emb_guesses_notes = [], []
    emb_preds_points, emb_preds_notes = [], []
    emb_preds_assets = []
    emb_preds_assets_notes = []
    point_labels = []
    note_labels = []
    chunk_tds = []
    for asset_point_label, asset_notes_label, (emb_scores_chunk_id, emb_scores_chunk_values) in tqdm(zip(asset_point_labels, asset_notes_labels, emb_scores_chunk.items())):
        point_labels.append(asset_point_label)
        note_labels.append(asset_notes_label)
        emb_guess_points_buffer = []
        emb_guess_notes_buffer = []
        close_chunk_keys = []
        apart_chunk_keys = []

        source_chunks = get_chunks_from_chunk_keys(filtered_doc, [emb_scores_chunk_id])
        for chunk_key, chunk_value in source_chunks.items():
            chunk_td = chunk_value['metadata']['timedelta']
            chunk_tds.append(chunk_td)
            # if(chunk_td<timedelta(10) and chunk_td>timedelta(-10)):
            #     close_chunk_keys.append(chunk_key)
            # else:
            #     apart_chunk_keys.append(chunk_key)
        # chunk_info = {}

        # chunk_info['timedeltas'] = chunk_tds
        # chunks_info[emb_scores_chunk_id] = chunk_info
        sorted_emb_scores = sorted(emb_scores_chunk_values.items(), key=lambda item: item[1], reverse=True)[:top_k]

        # print(sorted_emb_scores)
        top_results_emb = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_emb_scores]
        # print(top_results_emb)
        # print(top_results_emb)
        emb_results_guesses_points, emb_results_guesses_notes = infer_information(filtered_doc, top_results_emb)

        # print(emb_results_guesses_points)
        # print(emb_results_guesses_notes)
        

        emb_guesses_points.append(emb_results_guesses_points)
        emb_guess_points = max(emb_results_guesses_points, key=emb_results_guesses_points.get)
        emb_preds_points.append(emb_guess_points)
        emb_guess_points_buffer.append(emb_guess_points)
        emb_guesses_notes.append(emb_results_guesses_notes)
        # print(emb_results_guesses_notes)
        emb_guess_notes = max(emb_results_guesses_notes, key=emb_results_guesses_notes.get)
        emb_preds_notes.append(emb_guess_notes)
        emb_guess_notes_buffer.append(emb_guess_notes)
        
                
        emb_preds_assets.append(max(emb_guess_points_buffer))
        emb_preds_assets_notes.append(max(emb_guess_notes_buffer))
    # print(asset_point_label)
    # print(asset_notes_label)
        
    return chunk_tds, point_labels, emb_preds_points, note_labels, emb_preds_notes, asset_point_labels, emb_preds_assets, asset_notes_labels, emb_preds_assets_notes, emb_guesses_points, emb_guesses_notes
from tqdm import tqdm
def top_results_with_notes(filtered_doc, scores_chunk, top_k = 10):
    top_results_full = {}
    for (scores_chunk_id, scores_chunk_values) in scores_chunk.items():
        asset, point, note = get_hierarchy_from_chunk_key(filtered_doc, scores_chunk_id)
        
        source_chunks = get_chunks_from_chunk_keys(filtered_doc, [scores_chunk_id])
        for chunk_key, chunk_value in source_chunks.items():
            chunk_td = chunk_value['metadata']['timedelta']
        sorted_scores = sorted(scores_chunk_values.items(), key=lambda item: item[1], reverse=True)[:top_k]
        top_results_emb = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]
        assets, points, notes, tds = get_hierarchy(filtered_doc, top_results_emb)
        top_results_slice = [((doc_id, chunk_id), score, asset, point, note, td) for (((doc_id, chunk_id), score), asset, point, note, td) in zip(sorted_scores, assets, points, notes, tds)]
        
        top_results_full[scores_chunk_id] = {'chunk time delta': chunk_td, 'SIGRAG chunks': top_results_slice}
    return top_results_full

def top_results_without_tds(filtered_doc, scores_chunk, top_k = 10):
    top_results_full = {}
    for (scores_chunk_id, scores_chunk_values) in scores_chunk.items():
        asset, point, note = get_hierarchy_from_chunk_key(filtered_doc, scores_chunk_id)
        
        source_chunks = get_chunks_from_chunk_keys(filtered_doc, [scores_chunk_id])
        
        for chunk_key, chunk_value in source_chunks.items():
            chunk_date = chunk_value['metadata']['dates']
        sorted_scores = sorted(scores_chunk_values.items(), key=lambda item: item[1], reverse=True)[:top_k]
        top_results_emb = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]
        assets, points, notes, tds = get_hierarchy(filtered_doc, top_results_emb)
        top_results_slice = [((doc_id, chunk_id), score, asset, point, note, td) for (((doc_id, chunk_id), score), asset, point, note, td) in zip(sorted_scores, assets, points, notes, tds)]
        
        top_results_full[scores_chunk_id] = {'chunk date': chunk_date, 'SIGRAG chunks': top_results_slice}
    return top_results_full

def top_results_short(filtered_doc, scores_chunk, top_k = 10):
    top_results_full = []
    for (scores_chunk_id, scores_chunk_values) in scores_chunk.items():
        asset, point, note = get_hierarchy_from_chunk_key(filtered_doc, scores_chunk_id)
        
        source_chunks = get_chunks_from_chunk_keys(filtered_doc, [scores_chunk_id])
        
        for chunk_key, chunk_value in source_chunks.items():
            chunk_date = chunk_value['metadata']['dates']
        sorted_scores = sorted(scores_chunk_values.items(), key=lambda item: item[1], reverse=True)[:top_k]
        top_results_emb = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]
        assets, points, notes, tds = get_hierarchy(filtered_doc, top_results_emb)
        top_results_slice = [(asset, point, note, td) for (((doc_id, chunk_id), score), asset, point, note, td) in zip(sorted_scores, assets, points, notes, tds)]
        
        top_results_full.append({'chunk date': chunk_date, 'SIGRAG chunks': top_results_slice})
    return top_results_full


def point_level_predictions(filtered_doc, scores_chunk, point_labels, note_labels, top_k = 10):
    from tqdm import tqdm
    asset_point_labels = {}
    asset_note_labels = {}

    preds_points, preds_notes = {}, {}
    # preds_assets = []
    # preds_assets_notes = []
    chunk_tds = {}
    chunk_ids = {}
    top_score_tds = {}
    
    
    for (scores_chunk_id, scores_chunk_values), point_label, note_label in zip(scores_chunk.items(), point_labels, note_labels):
        chunk_doc_id, _ = get_keys_from_chunk_keys(filtered_doc, scores_chunk_id)

        if chunk_doc_id not in asset_point_labels:
            asset_point_labels[chunk_doc_id] = point_label 
        if chunk_doc_id not in asset_note_labels:
            asset_note_labels[chunk_doc_id] = note_label

            
        if chunk_doc_id in chunk_ids:
            chunk_ids[chunk_doc_id].append(scores_chunk_id)
        else:
            chunk_ids[chunk_doc_id]=[scores_chunk_id]
            
        source_chunks = get_chunks_from_chunk_keys(filtered_doc, [scores_chunk_id])
        for chunk_key, chunk_value in source_chunks.items():
            chunk_td = chunk_value['metadata']['timedelta']
            
        if chunk_doc_id in chunk_tds:
            chunk_tds[chunk_doc_id].append(chunk_td)
        else:
            chunk_tds[chunk_doc_id]=[chunk_td]

        sorted_scores = sorted(scores_chunk_values.items(), key=lambda item: item[1], reverse=True)[:top_k]
        top_results_emb = [(doc_id, chunk_id, score) for ((doc_id, chunk_id), score) in sorted_scores]
        top_spectra, top_levels, top_biases, top_speeds, top_recording_tds = get_recording(filtered_doc, top_results_emb, top_k = 10)
        
        results_guesses_points, results_guesses_notes = infer_information(filtered_doc, top_results_emb)
        
        guess_points = max(results_guesses_points, key=results_guesses_points.get)
        if chunk_doc_id in preds_points:
            preds_points[chunk_doc_id].append(guess_points)
        else:
            preds_points[chunk_doc_id]=[guess_points]
        guess_notes = max(results_guesses_notes, key=results_guesses_notes.get)
        if chunk_doc_id in preds_notes:
            preds_notes[chunk_doc_id].append(guess_notes)
        else:
            preds_notes[chunk_doc_id]=[guess_notes]
            
        if chunk_doc_id in top_score_tds:
            top_score_tds[chunk_doc_id].append(top_recording_tds)
        else:
            top_score_tds[chunk_doc_id]=[top_recording_tds]
        
    return chunk_ids, chunk_tds, preds_points, preds_notes, asset_point_labels, asset_note_labels, top_score_tds

def asset_level_predictions(filtered_doc, chunk_ids, chunk_tds, preds_points, preds_notes, before=50, after=20, top_k = 10):
    from tqdm import tqdm

    preds_assets_points = {}
    preds_assets_notes = {}

    for doc_id, chunk_tds_list, preds_points_list, preds_notes_list in zip(chunk_ids.keys(), chunk_tds.values(), preds_points.values(), preds_notes.values()):
        # Here we have lists of values associated with each asset
        # The goal is to use the relevant values for asset predictions
        # e.g., if we're using only recordings before the note, include only preds where td<0
        point_preds_buffer = []
        note_preds_buffer = []
        for chunk_td, preds_point, preds_note in zip(chunk_tds_list, preds_points_list, preds_notes_list):
            if chunk_td>timedelta(-before) and chunk_td<timedelta(after):
                point_preds_buffer.append(preds_point)
                note_preds_buffer.append(preds_note)
        if any(point_preds_buffer):
            preds_assets_point = max(point_preds_buffer)
            preds_assets_note = max(note_preds_buffer)
            preds_assets_points[doc_id]=preds_assets_point
            preds_assets_notes[doc_id]=preds_assets_note
        else:
            preds_assets_points[doc_id]=0
            preds_assets_notes[doc_id]=0
    return preds_assets_points, preds_assets_notes



from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
def compute_slice_preds(chunk_tds, point_labels, emb_preds_points, note_labels, emb_preds_notes, start = 50, stop = 20):
    point_slice_preds = [[] for i in range(start+stop)]
    point_slice_labels = [[] for i in range(start+stop)]
    note_slice_preds = [[] for i in range(start+stop)]
    note_slice_labels = [[] for i in range(start+stop)]
    for chunk_td, point_label, emb_preds_point, note_label, emb_preds_note in zip(chunk_tds, point_labels, emb_preds_points, note_labels, emb_preds_notes):
        for i in range(-start,stop):
            if chunk_td<timedelta(i+1) and chunk_td>timedelta(i):
                point_slice_labels[i+start].append(point_label)
                point_slice_preds[i+start].append(emb_preds_point)
                note_slice_labels[i+start].append(note_label)
                note_slice_preds[i+start].append(emb_preds_note)
            
    return point_slice_preds, point_slice_labels, note_slice_preds, note_slice_labels
def compute_slice_accuracies(chunk_tds, test_point_labels, test_emb_preds_points, test_note_labels, test_emb_preds_notes, query_emb_scores):
    
    before = 50
    after = 20
    slice_note_accuracies = []
    point_labels = np.zeros((before+after, ))
    point_preds = np.zeros((before+after))
    note_labels = np.zeros((before+after))
    note_preds = np.zeros((before+after))
    asset_point_labels = np.zeros((before+after))
    asset_point_preds = np.zeros((before+after))
    asset_note_labels = np.zeros((before+after))
    asset_note_preds = np.zeros((before+after))
    for i in range(-before,after):
        in_td_slice_point_labels = []
        in_td_slice_point_preds = []
        in_td_slice_note_labels = []
        in_td_slice_note_preds = []
        for chunk_td, test_point_label, test_emb_preds_point, test_note_label, test_emb_preds_note, (query_emb_key, query_emb_score) in zip(chunk_tds, test_point_labels, test_emb_preds_points, test_note_labels, test_emb_preds_notes, query_emb_scores.items()):
            if (chunk_td<timedelta(i) and chunk_td>timedelta(i-1)):
                in_td_slice_point_labels.append(test_point_label)
                in_td_slice_point_preds.append(test_emb_preds_point)
                in_td_slice_note_labels.append(test_note_label)
                in_td_slice_note_preds.append(test_emb_preds_note)
        point_labels.extend(in_td_slice_point_labels)
        point_preds.extend(in_td_slice_point_preds)
        note_labels.extend(in_td_slice_note_labels)
        note_preds.extend(in_td_slice_note_preds)
        
        asset_point_labels.append(max(in_td_slice_point_labels))
        asset_point_preds.append(max(in_td_slice_point_preds))
        asset_note_labels.append(max(in_td_slice_note_labels))
        asset_note_preds.append(max(in_td_slice_note_preds))
        
        slice_note_accuracies.append(balanced_accuracy_score(in_td_slice_note_labels, in_td_slice_note_preds))

def make_cnf_mat(labels, preds, display_labels = False, fig_x = 10, fig_y = 10, savefig = False, normalize = None):
    from sklearn import metrics
    import matplotlib.pyplot as plt
    emb_confusion_matrix = metrics.confusion_matrix(labels, preds, normalize = normalize)
    if any(display_labels):
        cnf_labels = display_labels
    else:
        cnf_labels = np.unique(labels)
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = emb_confusion_matrix, display_labels = cnf_labels)
    fig, ax = plt.subplots(figsize=(fig_x, fig_y))
    cm_display.plot(ax=ax)
    if savefig:
        fig.savefig(savefig, bbox_inches='tight')
    plt.show()

import pickle
def SIGRAG_test_loop(vector_store, start = 50, stop = 20, top_score_tds = {}, 
                     all_point_slice_preds = [], all_point_slice_labels = [], all_note_slice_preds = [], all_note_slice_labels = [],
                     all_point_asset_preds = [], all_point_asset_labels = [], all_note_asset_preds = [], all_note_asset_labels = [],
                     all_point_slice_preds_file = 'all_point_slice_preds.pkl', all_point_slice_labels_file = 'all_point_slice_labels.pkl',
                     all_note_slice_preds_file = 'all_note_slice_preds.pkl', all_note_slice_labels_file = 'all_note_slice_labels.pkl'):
    
    all_point_slice_preds_file = 'excluded_point_slice_preds.pkl'
    all_point_slice_labels_file = 'excluded_point_slice_labels.pkl'
    all_note_slice_preds_file = 'excluded_note_slice_preds.pkl'
    all_note_slice_labels_file = 'included_note_slice_labels.pkl'
    
    all_point_asset_preds_file = 'excluded_point_asset_preds.pkl'
    all_point_asset_labels_file = 'excluded_point_asset_labels.pkl'
    all_note_asset_preds_file = 'excluded_note_asset_preds.pkl'
    all_note_asset_labels_file = 'excluded_note_asset_labels.pkl'
    before = start
    after = stop
    doc_counter = 0
    
    all_point_asset_preds = [[] for i in range(start+stop)]
    all_point_asset_labels = [[] for i in range(start+stop)]
    all_note_asset_preds = [[] for i in range(start+stop)]
    all_note_asset_labels = [[] for i in range(start+stop)]
    
    for doc_id, doc in tqdm(vector_store.items()):
    
        doc_chunks = doc['recording_chunks']
        chunk_keys = list(doc_chunks.keys())
        if any(chunk_keys):
            asset = doc['asset']
            points = doc['point']
            pointname = points['Name']
            note = doc['note']
            notecontent = note['noteComment']
            
            emb_scores, asset_point_labels, asset_notes_labels = compute_embedding_distance_final(vector_store, chunk_keys, exclude_same_point = True)
            chunk_ids, chunk_tds, preds_points, preds_notes, preds_point_labels, preds_note_labels, top_score_td = point_level_predictions(
                vector_store, emb_scores, asset_point_labels, asset_notes_labels)
            top_score_tds[doc_id] = top_score_td
        
            flattened_points_preds = []
            flattened_notes_preds = []
            flattened_td_preds = []
            flattened_id_preds = []
            for preds_points_keys, preds_points_values in preds_points.items():
                flattened_points_preds.extend(preds_points_values)
            for preds_notes_keys, preds_notes_values in preds_notes.items():
                flattened_notes_preds.extend(preds_notes_values)
            for chunk_tds_keys, chunk_tds_values in chunk_tds.items():
                flattened_td_preds.extend(chunk_tds_values)
            for chunk_ids_keys, chunk_ids_values in chunk_ids.items():
                flattened_id_preds.extend(chunk_ids_values)
                
            point_slice_preds, point_slice_labels, note_slice_preds, note_slice_labels = compute_slice_preds(
            flattened_td_preds, asset_point_labels, flattened_points_preds, asset_notes_labels, flattened_notes_preds, start, stop)
            all_point_slice_preds.append(point_slice_preds)
            all_point_slice_labels.append(point_slice_labels)
            all_note_slice_preds.append(note_slice_preds)
            all_note_slice_labels.append(note_slice_labels)
            
            assets_points_labels = list(preds_point_labels.values())
            assets_notes_labels = list(preds_note_labels.values())
            
            # for i in range(0,after):
            #     for j in range(0, before):
            #         if i or j:
            #             preds_assets_points, preds_assets_notes = asset_level_predictions(vector_store, chunk_ids, chunk_tds, preds_points, preds_notes, before=j, after=i, top_k = 10)
            #             if any(preds_assets_points.values()):
            #                 #assets_points_slices[i].append(balanced_accuracy_score(assets_points_labels, list(preds_assets_points.values())))
            #                 all_point_asset_preds[i][j].append(list(preds_assets_points.values()))
            #                 all_point_asset_labels[i][j].append(assets_points_labels)
        
            #             if any(preds_assets_notes.values()):    
            #                 #assets_notes_slices[i].append(balanced_accuracy_score(assets_notes_labels, list(preds_assets_notes.values())))
            #                 all_note_asset_preds[i][j].append(list(preds_assets_notes.values()))
            #                 all_note_asset_labels[i][j].append(assets_notes_labels)
            if not doc_counter%50 and doc_counter:
                with open(all_point_slice_preds_file, "wb") as f:
                    pickle.dump(all_point_slice_preds, f)
                with open(all_point_slice_labels_file, "wb") as g:
                    pickle.dump(all_point_slice_labels, g)
                with open(all_note_slice_preds_file, "wb") as h:
                    pickle.dump(all_note_slice_preds, h)
                with open(all_note_slice_labels_file, "wb") as a:
                    pickle.dump(all_note_slice_labels, a)
                # with open(all_point_asset_preds_file, "wb") as f:
                #     pickle.dump(all_point_asset_preds, f)
                # with open(all_point_asset_labels_file, "wb") as g:
                #     pickle.dump(all_point_asset_labels, g)
                # with open(all_note_asset_preds_file, "wb") as h:
                #     pickle.dump(all_note_asset_preds, h)
                # with open(all_note_asset_labels_file, "wb") as a:
                #     pickle.dump(all_note_asset_labels, a)    
            doc_counter+=1

    return top_score_tds, all_point_slice_preds, all_point_slice_labels, all_note_slice_preds , all_note_slice_labels, all_point_asset_preds, all_point_asset_labels, all_note_asset_preds, all_note_asset_labels
