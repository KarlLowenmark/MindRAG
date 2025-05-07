# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 14:45:23 2025

@author: karle
"""
import os, sys
cwd = os.getcwd()
sys.path.append(os.path.join(os.path.dirname(file_path)))
from importlib import reload
from langchain_core.messages import SystemMessage, HumanMessage
from IPython.display import Image
import LLCMM_support_functions
import warnings
from sklearn.exceptions import DataConversionWarning
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import json
import pickle
import SIGRAG
import warnings
import time
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import SIGRAG_test
import copy
warnings.filterwarnings(action='ignore', category=UserWarning)

reload(SIGRAG)
reload(SIGRAG_test)


async def test_SIGRAG_agent(vector_store, source_vector_store, filtered_vector_store, prediction_agent = None, evaluation_agent = None, using_recordings = False):
    if not prediction_agent:
        prediction_agent = SIGRAG_test.set_up_prediction_agent()
    if not evaluation_agent:
        evaluation_agent = SIGRAG_test.set_up_evaluation_agent()

    made_pred_and_eval = False
    filtered_asset_chunks = []
    prev_asset_path = ""
    prev_point_names = []
    point_types = []
    prev_note = []
    note_labels = [["kabel", "giv", "sens"], ["bpfo", "bpfi", "bsf"], ["glapp", "haveri", "obalans"]]
    note_labels = [['bromsgenerator']]
    have_analysed_counter = 0
    #Image(prediction_agent.get_graph().draw_png())
    
    for doc_nr, (doc_id, doc) in enumerate(source_vector_store.items()):
        doc_chunks = doc['recording_chunks']
        filtered_chunk_keys = list(doc_chunks.keys())
        if any(filtered_chunk_keys):
            filtered_asset = doc['asset']
            filtered_points = doc['point']
            filtered_pointtype = filtered_points['DetectionName']
            filtered_pointname = filtered_points['Name']
            filtered_note = doc['note']
            filtered_notecomment = filtered_note['noteComment']
            filtered_idnote = filtered_note['idNote']

            if True:
                filtered_chunks = SIGRAG.get_chunks_from_chunk_keys(source_vector_store, filtered_chunk_keys)
                chunk_doc_id, _ = SIGRAG.get_keys_from_chunk_keys(source_vector_store, filtered_chunk_keys[0])
                filtered_emb_scores, filtered_asset_point_labels, filtered_asset_notes_labels = SIGRAG.compute_embedding_distance_final(
                    filtered_vector_store, filtered_chunk_keys, filtered_chunks, chunk_doc_id, exclude_same_point = True)
                # filtered_input_chunk_key = list(filtered_emb_scores.keys())[0]
                # filtered_top_results_with_notes = SIGRAG.top_results_without_tds(vector_store, filtered_emb_scores, top_k = 5)
                filtered_top_results_with_notes = SIGRAG.top_results_short(vector_store, filtered_emb_scores, top_k = 5)
                # filtered_agent_input_asset, filtered_agent_input_point, filtered_agent_input_note = SIGRAG.get_hierarchy_from_chunk_key(vector_store, filtered_input_chunk_key)
                
                filtered_input_path = " ".join(filtered_asset['Path'].split("\\")[2:])
                
                filtered_input_name = filtered_points['Name']
                filtered_input_date = filtered_note['noteDateUTC']
                filtered_input_note = filtered_note['noteComment']
                print(filtered_input_note)
                print(filtered_input_path)
                print(filtered_pointname)
                # filtered_input_chunks = list(reversed(list(filtered_top_results_with_notes.items())))
                filtered_input_chunks = list(reversed(filtered_top_results_with_notes))
                if not doc_nr:
                    prev_idnote = filtered_idnote
                if prev_idnote == filtered_idnote and doc_nr != (len(source_vector_store)-1):
                    filtered_asset_chunks.append(filtered_input_chunks)
                    point_types.append(filtered_pointtype)
                    prev_asset_path = filtered_input_path
                    prev_point_names.append(filtered_pointname)
                    prev_note = filtered_note
                else:
                    input_dict = [{} for i in range(len(filtered_asset_chunks))]
                    max_rec_len = 0
                    for filtered_point_chunks in filtered_asset_chunks:
                        rec_len = len(filtered_point_chunks)
                        if rec_len > max_rec_len:
                            max_rec_len = rec_len
                    input_chunk_list = [[] for i in range(max_rec_len)]
                    for point_nr, (filtered_point_chunks, point_type, prev_point_name) in enumerate(zip(filtered_asset_chunks, point_types, prev_point_names)):
                        input_dict[point_nr]['sensor type'] = point_type
                        input_dict[point_nr]['point name'] = prev_point_name
                        for filtered_point_chunk_nr, filtered_point_chunk in enumerate(filtered_point_chunks):
                            input_chunk_list[filtered_point_chunk_nr].append(filtered_point_chunk)
                    input_string = f"""Your job is to make fault diagnosis predictions based on data from a condition monitoring (CM) database.
                    You are currently analysing signals from the asset at machine path {prev_asset_path}, with point names
                    and sensor types {input_dict}, where the positions of the point names and sensor types correspond to the positions of the
                    incoming recording slices.
                    The recording slices will feature one set of recordings per point name and sensor type, and iterate forward in time based on your requests.
                    Each recording features the datetime value of a recording from the machine you're analysing, 
                    and five SIGRAG chunks, that are obtained by finding the most similar recordings from the CM database.
                    These SIGRAG chunks have associated annotations that describe fault properties of these chunks, and time deltas which describe the time
                    distance between the SIGRAG chunk recording and its associated annotation.
                    
                    If you are not ready to make a prediction, reply with "CONTINUE" and the user will provide another set of recording slices.
                    If you ask for more slices than are available, the user will break the loop and you may proceed with your analysis.
                    
                    Finally, respond with your analysis with the "reply_with_precition" tool.
                    
                    The first input and SIGRAG chunks are: {input_chunk_list[0]}
                    """
                    print(f"Starting agent chain for the points {prev_point_names} with the note {filtered_note['noteComment']}")
                    break_key = False
                    input_chunk_slices = []
                    for input_chunk_slice_nr, input_chunk_slice in enumerate(input_chunk_list):
                        input_chunk_slices.append(input_chunk_slice)
                        call_agent_flag = False
                        if any(input_chunk_slices):
                            recording_date_to_write = datetime.strptime(input_chunk_slice[0]['chunk date'][0:19], '%Y-%m-%dT%H:%M:%S')
                            note_date_to_write = datetime.strptime(prev_note['noteDateUTC'][0:19], '%Y-%m-%dT%H:%M:%S')
                            days_between_note_and_recording = (recording_date_to_write - note_date_to_write).days
                            if not input_chunk_slice_nr:
                                prediction_agent_input = input_string
                                call_agent_flag = True
                            elif not input_chunk_slice_nr % 40:
                                prediction_agent_input = f"""The next slices are {input_chunk_slices}"""
                                call_agent_flag = True
                            if call_agent_flag:
                                async for output in prediction_agent.astream({
                                    "input": prediction_agent_input,
                                    "chat_history": [],
                                    }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
                                    # stream_mode="updates" yields dictionaries with output keyed by node name
                                    for key, value in output.items():
                                        
                                        if key == "exit_graph":
                                            print(f"Output from node '{key}':")
                                            print("---")
                                            print(value["intermediate_steps"][-1])
                                            do_nothing = True
                                        elif key == "prediction_thinker":
                                            print(f"Output from node '{key}':")
                                            print("---")
                                            print(value["intermediate_steps"][-1])
                                            do_nothing = True
                                        elif key == "get_signal":
                                            break_key = True
                                            print(f"Output from node '{key}':")
                                            print("---")
                                            print(value["intermediate_steps"][-1])
                                            break
                                        elif key == "reply_with_prediction":
                                            break_key = True
                                            print(f"Output from node '{key}':")
                                            print("---")
                                            print(value["intermediate_steps"][-1])
                                            break
                                        else:
                                            print(f"Wrong tool {key} called in this part of the loop.")
                                            break
                            if call_agent_flag:
                                input_chunk_slices = []
                                        
                                    # print("\n---\n")
                                
                        if break_key:
                            break
                        # if days_between_note_and_recording<0:
                        #     break
                    stupid_AI_counter = 0
                    while not break_key:
                        print(f"Something went wrong. {input_chunk_slice_nr} slices analysed and {days_between_note_and_recording} days until note.")
                        async for output in prediction_agent.astream({
                            "input": """You have run out of chunks. You MUST make a prediction with what you have using "reply_with_prediction" or you will not progress. You will not
                            get any more chunks. Proceed through "reply_with_prediction", and if you have not detected any faults, please explain why.""",
                            "chat_history": [],
                            }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
                            for key, value in output.items():
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                        if 'fault_diagnosis_prediction' in value['intermediate_steps'][0].tool_input:
                            break_key = True
                            break
                        else:
                            time.sleep(10)
                            stupid_AI_counter+=1
                        if stupid_AI_counter>5:
                            break
                            
                    
                    agent_prediction = value['intermediate_steps'][0].tool_input['fault_diagnosis_prediction']
                    input_chunk_ids = []
                    input_recordings = []
                    for input_chunk_element in input_chunk_slice:
                        input_chunk_ids.append(input_chunk_element[0])
                        input_recording = SIGRAG.get_recording_from_chunk_key(vector_store, input_chunk_element[0])
                        if any(input_recording):
                            input_recordings.append(input_recording[0])
                        else:
                            print(f"Error! No recording at {input_chunk_element[0]}")
        
                    replied_with_prediction = False
                    
                    if using_recordings:
                        prediction_input = f"""Ok. The most recent input slice is {input_recordings}, ordered the same as the SIGRAG chunks per the information in
                                {input_dict}."""
                        async for output in prediction_agent.astream({
                            "input": prediction_input,
                            "chat_history": [],
                            }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
                            # stream_mode="updates" yields dictionaries with output keyed by node name
                            for key, value in output.items():
                                if key == "reply_with_prediction":
                                    replied_with_prediction = True
                                else:
                                    prediction_input = f"""You must reply with your prediction before proceeding."""
                                #print(f"Output from node '{key}':")
                                #print("---")
                                #print(value["intermediate_steps"][-1])
                            print("\n---\n")
    
                    
        
                    async for output in prediction_agent.astream({
                        "input": f"""Based on the SIGRAG chunks and recordings, what is your fault assessment?""",
                        "chat_history": [],
                        }
                        , config = {"configurable": {"thread_id": str(doc_nr)}}
                        , stream_mode="updates"):
                        # stream_mode="updates" yields dictionaries with output keyed by node name
                        #for key, value in output.items():
                            #if key == "reply_with_prediction":
                            #   replied_with_prediction = True
                            #else:
                            #    prediction_input = 
                            #print(f"Output from node '{key}':")
                            #print("---")
                            #print(value["intermediate_steps"][-1])
                        print("\n---\n")
                    
                    written_flag = False
                    written_counter = 0
                    
                    #while not written_flag:
                    
                    async for output in evaluation_agent.astream({
                        "input": f"""The true note is {prev_note['noteComment']} at a time delta of {days_between_note_and_recording} days. The prediction made was:
                        {agent_prediction}. Was the analysis correct? Write it to file with the "write_prediction" tool.
                        Remember to write the asset path {prev_asset_path} with point name and sensor types {input_dict} to file.""",
                        "chat_history": [],
                        }
                            , config = {"configurable": {"thread_id": str(doc_nr)}}
                            , stream_mode="updates"):
                        # stream_mode="updates" yields dictionaries with output keyed by node name
                        for key, value in output.items():
                            if key == "write_prediction":
                                written_flag=True
                                print("Writing prediction to file!")
                            else:
                                written_counter+=1
                                if written_counter>5:
                                    written_flag=True
                            print(f"Output from node '{key}':")
                            print("---")
                            print(value["intermediate_steps"][-1])
                        print("\n---\n")
                    
                    filtered_asset_chunks = [] 
                    filtered_asset_chunks.append(filtered_input_chunks)
                    point_types = []
                    point_types.append(filtered_pointtype)
                    prev_asset_path = filtered_input_path
                    prev_point_names = []
                    prev_point_names.append(filtered_pointname)
                    prev_idnote = filtered_idnote
                    prev_note = filtered_note
                    made_pred_and_eval = True
               
        if have_analysed_counter > 5:
            break
        
async def test_SIGRAG_agent_parts(vector_store, source_vector_store, filtered_vector_store, prediction_agent = None, evaluation_agent = None, using_recordings = False):
    if not prediction_agent:
        prediction_agent = SIGRAG_test.set_up_prediction_agent()
    if not evaluation_agent:
        evaluation_agent = SIGRAG_test.set_up_evaluation_agent()

    made_pred_and_eval = False
    filtered_asset_chunks = []
    prev_asset_path = ""
    prev_point_names = []
    point_types = []
    prev_note = []
    note_labels = [["kabel", "giv", "sens"], ["bpfo", "bpfi", "bsf"], ["glapp", "haveri", "obalans"]]
    # note_labels = [['bromsgenerator']]
    have_analysed_counter = 0
    #Image(prediction_agent.get_graph().draw_png())
    second_iteration = False
    note_flag = False
    for doc_nr, (doc_id, doc) in enumerate(source_vector_store.items()):
        prev_note_flag = False
        doc_chunks = doc['recording_chunks']
        filtered_chunk_keys = list(doc_chunks.keys())
        filtered_asset = doc['asset']
        filtered_path = " ".join(filtered_asset['Path'].split("\\")[2:])
        filtered_points = doc['point']
        filtered_pointtype = filtered_points['DetectionName']
        filtered_pointname = filtered_points['Name']
        filtered_note = doc['note']
        filtered_notecomment = filtered_note['noteComment']
        filtered_idnote = filtered_note['idNote']
        if not doc_nr:
            prev_idnote = filtered_idnote
        # if any(filtered_chunk_keys):
        if note_flag:
            prev_note_flag = True
        note_flag = False
        for note_label in note_labels:
            for note_lab in note_label:
                if note_lab.lower() in filtered_notecomment.lower():
                    note_flag = True

        print(filtered_notecomment)
        print(filtered_pointname)
        print(filtered_path)
        if note_flag:
            filtered_chunks = SIGRAG.get_chunks_from_chunk_keys(source_vector_store, filtered_chunk_keys)
            chunk_doc_id, _ = SIGRAG.get_keys_from_chunk_keys(source_vector_store, filtered_chunk_keys[0])
            filtered_emb_scores, filtered_asset_point_labels, filtered_asset_notes_labels = SIGRAG.compute_embedding_distance_final(
                filtered_vector_store, filtered_chunk_keys, filtered_chunks, chunk_doc_id, exclude_same_point = True)
            # filtered_input_chunk_key = list(filtered_emb_scores.keys())[0]
            filtered_top_results_with_notes = SIGRAG.top_results_short(vector_store, filtered_emb_scores, top_k = 5)
            # filtered_agent_input_asset, filtered_agent_input_point, filtered_agent_input_note = SIGRAG.get_hierarchy_from_chunk_key(vector_store, filtered_input_chunk_key)
            
            # filtered_input_path = " ".join(filtered_agent_input_asset['Path'].split("\\")[2:])
            
            # filtered_input_name = filtered_agent_input_point['Name']
            # filtered_input_date = filtered_agent_input_note['noteDateUTC']
            # filtered_input_note = filtered_agent_input_note['noteComment']
            
            # filtered_input_chunks = list(reversed(list(filtered_top_results_with_notes.items())))
            filtered_input_chunks_input = list(reversed(filtered_top_results_with_notes))
            # return filtered_input_chunks
            
        if note_flag:# and prev_idnote == filtered_idnote and doc_nr != (len(source_vector_store)-1):
            filtered_asset_chunks.append(filtered_input_chunks_input)
            point_types.append(filtered_pointtype)
            prev_asset_path = filtered_path
            prev_point_names.append(filtered_pointname)
            prev_note = filtered_note
        elif prev_note_flag:
            input_dict = [{} for i in range(len(filtered_asset_chunks))]
            max_rec_len = 0
            for filtered_point_chunks in filtered_asset_chunks:
                rec_len = len(filtered_point_chunks)
                if rec_len > max_rec_len:
                    max_rec_len = rec_len
            input_chunk_list = [[] for i in range(max_rec_len)]
            for point_nr, (filtered_point_chunks, point_type, prev_point_name) in enumerate(zip(filtered_asset_chunks, point_types, prev_point_names)):
                input_dict[point_nr]['sensor type'] = point_type
                input_dict[point_nr]['point name'] = prev_point_name
                for filtered_point_chunk_nr, filtered_point_chunk in enumerate(filtered_point_chunks):
                    input_chunk_list[filtered_point_chunk_nr].append(filtered_point_chunk)
            input_string = f"""Your job is to make fault diagnosis predictions based on data from a condition monitoring (CM) database.
            You are currently analysing signals from the asset at machine path {prev_asset_path}, with point names
            and sensor types {input_dict}, where the positions of the point names and sensor types correspond to the positions of the
            incoming recording slices.
            The recording slices will feature one set of recordings per point name and sensor type, and iterate forward in time based on your requests.
            Each recording features a chunk id and datetime value at the start of the list, which is the input chunk id with the date of the recording.
            Each input chunk is associated with five SIGRAG chunks, that are obtained by finding the most similar recordings from the CM database.
            These SIGRAG chunks have associated annotations that describe fault properties of these chunks, and time deltas which describe the time
            distance between the SIGRAG chunk and its annotation.
            
            If you are not ready to make a prediction, reply with "CONTINUE" and the user will provide another set of recording slices.
            If you ask for more slices than are available, the user will break the loop and you may proceed with your analysis.
            
            Finally, respond with your analysis with the "reply_with_precition" tool.
            
            The first input and SIGRAG chunks are: {input_chunk_list[0]}
            """
            print(f"Starting agent chain for the points {prev_point_names} with the note {prev_note['noteComment']}")
            break_key = False
            input_chunk_slices = []
            for input_chunk_slice_nr, input_chunk_slice in enumerate(input_chunk_list):
                
                input_chunk_slice_nr += 1
                
                input_chunk_slices.append(input_chunk_slice)
                
                if any(input_chunk_slices):
                    recording_date_to_write = datetime.strptime(input_chunk_slice[-1]['chunk date'][0:19], '%Y-%m-%dT%H:%M:%S')
                    note_date_to_write = datetime.strptime(prev_note['noteDateUTC'][0:19], '%Y-%m-%dT%H:%M:%S')
                    days_between_note_and_recording = (recording_date_to_write - note_date_to_write).days
                    if not input_chunk_slice_nr:
                        prediction_agent_input = input_string
                        call_agent_flag = True
                    elif not input_chunk_slice_nr % 40:
                        agent_input_slices = copy.deepcopy(input_chunk_slices)
                        prediction_agent_input = f"""The next slices are {agent_input_slices}"""
                        call_agent_flag = True
                        if second_iteration:
                            return prediction_agent_input
                        input_chunk_slices = []
                    else:
                        
                        call_agent_flag = False
                    if call_agent_flag:
                        # async for output in prediction_agent.astream({
                        #     "input": prediction_agent_input,
                        #     "chat_history": [],
                        #     }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
                        #     # stream_mode="updates" yields dictionaries with output keyed by node name
                        #     for key, value in output.items():
                                
                        #         if key == "exit_graph":
                        #             print(f"Output from node '{key}':")
                        #             print("---")
                        #             print(value["intermediate_steps"][-1])
                        #             do_nothing = True
                        #         elif key == "prediction_thinker":
                        #             print(f"Output from node '{key}':")
                        #             print("---")
                        #             print(value["intermediate_steps"][-1])
                        #             do_nothing = True
                        #         elif key == "get_signal":
                        #             break_key = True
                        #             print(f"Output from node '{key}':")
                        #             print("---")
                        #             print(value["intermediate_steps"][-1])
                        #             break
                        #         elif key == "reply_with_prediction":
                        #             break_key = True
                        #             print(f"Output from node '{key}':")
                        #             print("---")
                        #             print(value["intermediate_steps"][-1])
                        #             break
                        #         else:
                        #             print(f"Wrong tool {key} called in this part of the loop.")
                        #             break

                        input_chunk_slices = []
                                
                            # print("\n---\n")
                        
                if break_key:
                    break
                # if days_between_note_and_recording<0:
                #     break
            stupid_AI_counter = 0
            # while not break_key:
            #     print(f"Something went wrong. {input_chunk_slice_nr} slices analysed and {days_between_note_and_recording} days until note.")
            #     async for output in prediction_agent.astream({
            #         "input": """You have run out of chunks. You MUST make a prediction with what you have using "reply_with_prediction" or you will not progress. You will not
            #         get any more chunks. Proceed through "reply_with_prediction", and if you have not detected any faults, please explain why.""",
            #         "chat_history": [],
            #         }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #         for key, value in output.items():
            #             print(f"Output from node '{key}':")
            #             print("---")
            #             print(value["intermediate_steps"][-1])
            #     if 'fault_diagnosis_prediction' in value['intermediate_steps'][0].tool_input:
            #         break_key = True
            #         break
            #     else:
            #         time.sleep(10)
            #         stupid_AI_counter+=1
            #     if stupid_AI_counter>5:
            #         break
                    
            
            # agent_prediction = value['intermediate_steps'][0].tool_input['fault_diagnosis_prediction']
            
            # input_chunk_ids = []
            # input_recordings = []
            # for input_chunk_element in input_chunk_slice:
            #     input_chunk_ids.append(input_chunk_element[0])
            #     input_recording = SIGRAG.get_recording_from_chunk_key(vector_store, input_chunk_element[0])
            #     if any(input_recording):
            #         input_recordings.append(input_recording[0])
            #     else:
            #         print(f"Error! No recording at {input_chunk_element[0]}")

            # replied_with_prediction = False
            
            # if using_recordings:
            #     prediction_input = f"""Ok. The most recent input slice is {input_recordings}, ordered the same as the SIGRAG chunks per the information in
            #             {input_dict}."""
            #     async for output in prediction_agent.astream({
            #         "input": prediction_input,
            #         "chat_history": [],
            #         }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #         # stream_mode="updates" yields dictionaries with output keyed by node name
            #         for key, value in output.items():
            #             if key == "reply_with_prediction":
            #                 replied_with_prediction = True
            #             else:
            #                 prediction_input = f"""You must reply with your prediction before proceeding."""
            #             #print(f"Output from node '{key}':")
            #             #print("---")
            #             #print(value["intermediate_steps"][-1])
            #         print("\n---\n")

            

            # async for output in prediction_agent.astream({
            #     "input": f"""Based on the SIGRAG chunks and recordings, what is your fault assessment?""",
            #     "chat_history": [],
            #     }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #     # stream_mode="updates" yields dictionaries with output keyed by node name
            #     #for key, value in output.items():
            #         #if key == "reply_with_prediction":
            #         #   replied_with_prediction = True
            #         #else:
            #         #    prediction_input = 
            #         #print(f"Output from node '{key}':")
            #         #print("---")
            #         #print(value["intermediate_steps"][-1])
            #     print("\n---\n")
            
            written_flag = False
            written_counter = 0
            
            #while not written_flag:
            
            # async for output in evaluation_agent.astream({
            #     "input": f"""The true note is {prev_note['noteComment']} at a time delta of {days_between_note_and_recording} days. The prediction made was:
            #     {agent_prediction}. Was the analysis correct? Write it to file with the "write_prediction" tool.
            #     Remember to write the asset path {prev_asset_path} with point name and sensor types {input_dict} to file.""",
            #     "chat_history": [],
            #     }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #     # stream_mode="updates" yields dictionaries with output keyed by node name
            #     for key, value in output.items():
            #         if key == "write_prediction":
            #             written_flag=True
            #             print("Writing prediction to file!")
            #         else:
            #             written_counter+=1
            #             if written_counter>5:
            #                 written_flag=True
            #         print(f"Output from node '{key}':")
            #         print("---")
            #         print(value["intermediate_steps"][-1])
            #     print("\n---\n")
            
            filtered_asset_chunks = [] 
            filtered_asset_chunks.append(filtered_input_chunks_input)
            point_types = []
            point_types.append(filtered_pointtype)
            prev_asset_path = filtered_path
            prev_point_names = []
            prev_point_names.append(filtered_pointname)
            prev_idnote = filtered_idnote
            prev_note = filtered_note
            made_pred_and_eval = True
            have_analysed_counter +=1
            second_iteration = True
        
        
        
async def test_SIGRAG_agent_notefilter(vector_store, source_vector_store, filtered_vector_store, prediction_agent = None, evaluation_agent = None, using_recordings = False, step_size = 1, note_labels = False):
    if not prediction_agent:
        prediction_agent = SIGRAG_test.set_up_prediction_agent()
    if not evaluation_agent:
        evaluation_agent = SIGRAG_test.set_up_evaluation_agent()

    made_pred_and_eval = False
    filtered_asset_chunks = []
    prev_asset_path = ""
    prev_point_names = []
    point_types = []
    prev_note = []
    if not note_labels:
        note_labels = [["kabel", "giv", "sens"], ["bpfo", "bpfi", "bsf"], ["glapp", "haveri", "obalans"]]
    # note_labels = [['bromsgenerator']]
    have_analysed_counter = 0
    #Image(prediction_agent.get_graph().draw_png())
    note_flag = False
    for doc_nr, (doc_id, doc) in enumerate(source_vector_store.items()):
        prev_note_flag = False
        doc_chunks = doc['recording_chunks']
        filtered_chunk_keys = list(doc_chunks.keys())
        filtered_asset = doc['asset']
        filtered_path = " ".join(filtered_asset['Path'].split("\\")[2:])
        filtered_points = doc['point']
        filtered_pointtype = filtered_points['DetectionName']
        filtered_pointname = filtered_points['Name']
        filtered_note = doc['note']
        filtered_notecomment = filtered_note['noteComment']
        filtered_idnote = filtered_note['idNote']
        if not doc_nr:
            prev_idnote = filtered_idnote
        # if any(filtered_chunk_keys):
        if note_flag:
            prev_note_flag = True
        note_flag = False
        for note_label in note_labels:
            for note_lab in note_label:
                if note_lab.lower() in filtered_notecomment.lower():
                    note_flag = True

        print(filtered_notecomment)
        print(filtered_pointname)
        print(filtered_path)
        if note_flag:
            filtered_chunks = SIGRAG.get_chunks_from_chunk_keys(source_vector_store, filtered_chunk_keys)
            chunk_doc_id, _ = SIGRAG.get_keys_from_chunk_keys(source_vector_store, filtered_chunk_keys[0])
            filtered_emb_scores, filtered_asset_point_labels, filtered_asset_notes_labels = SIGRAG.compute_embedding_distance_final(
                filtered_vector_store, filtered_chunk_keys, filtered_chunks, chunk_doc_id, exclude_same_point = True)
            # filtered_input_chunk_key = list(filtered_emb_scores.keys())[0]
            filtered_top_results_with_notes = SIGRAG.top_results_short(vector_store, filtered_emb_scores, top_k = 5)
            # filtered_agent_input_asset, filtered_agent_input_point, filtered_agent_input_note = SIGRAG.get_hierarchy_from_chunk_key(vector_store, filtered_input_chunk_key)
            
            # filtered_input_path = " ".join(filtered_agent_input_asset['Path'].split("\\")[2:])
            
            # filtered_input_name = filtered_agent_input_point['Name']
            # filtered_input_date = filtered_agent_input_note['noteDateUTC']
            # filtered_input_note = filtered_agent_input_note['noteComment']
            
            # filtered_input_chunks = list(reversed(list(filtered_top_results_with_notes.items())))
            filtered_input_chunks_input = list(reversed(filtered_top_results_with_notes))
            # return filtered_input_chunks
            
        if note_flag and prev_idnote == filtered_idnote and doc_nr != (len(source_vector_store)-1):
            filtered_asset_chunks.append(filtered_input_chunks_input)
            point_types.append(filtered_pointtype)
            prev_asset_path = filtered_path
            prev_point_names.append(filtered_pointname)
            prev_note = filtered_note
        elif prev_note_flag:
            input_dict = [{} for i in range(len(filtered_asset_chunks))]
            input_SIGRAG_chunks = {}
            input_chunk_dict = {}
            max_rec_len = 0
            for filtered_point_chunks in filtered_asset_chunks:
                rec_len = len(filtered_point_chunks)
                if rec_len > max_rec_len:
                    max_rec_len = rec_len
            input_chunk_list = [[] for i in range(max_rec_len)]
            input_SIGRAG_chunks = [[] for i in range(max_rec_len)]
            for point_nr, (filtered_point_chunks, point_type, prev_point_name) in enumerate(zip(filtered_asset_chunks, point_types, prev_point_names)):
                input_dict[point_nr]['sensor type'] = point_type
                input_dict[point_nr]['point name'] = prev_point_name
                input_dict[point_nr]['SIGRAG chunks'] = filtered_point_chunks
                # input_SIGRAG_chunks[prev_point_name] = 
                for filtered_point_chunk_nr, filtered_point_chunk in enumerate(filtered_point_chunks):
                    # for SIGRAG_chunks in filtered_point_chunk['SIGRAG chunks']:
                    #     for SIGRAG_content in SIGRAG_chunks:
                    #         SIGRAG_content_keys = list(SIGRAG_content.keys())
                    #         SIGRAG_content_values = list(SIGRAG_content.values())
                            
                    #         for SIGRAG_content_key, SIGRAG_content_value in zip(SIGRAG_content_keys, SIGRAG_content_values):
                    #             if SIGRAG_content_key not in input_SIGRAG_chunks[prev_point_name].keys():
                    #                 input_SIGRAG_chunks[prev_point_name][SIGRAG_content_key] = []
                    #             input_SIGRAG_chunks[prev_point_name][SIGRAG_content_key].append(SIGRAG_content_value)
                    new_SIGRAG_chunks = {}
                    recording_chunk_date = filtered_point_chunk['chunk date']
                    for SIGRAG_chunk in filtered_point_chunk['SIGRAG chunks']:
                        for SIGRAG_content in SIGRAG_chunk:
                            #print(list(SIGRAG_content.items()))
                            SIGRAG_content_keys = list(SIGRAG_content.keys())
                            SIGRAG_content_values = list(SIGRAG_content.values())
                            
                            for SIGRAG_content_key, SIGRAG_content_value in zip(SIGRAG_content_keys, SIGRAG_content_values):
                                print(SIGRAG_content_key)
                                if SIGRAG_content_key not in new_SIGRAG_chunks:
                                    new_SIGRAG_chunks[SIGRAG_content_key] = []
                                new_SIGRAG_chunks[SIGRAG_content_key].append(SIGRAG_content_value)

                    input_SIGRAG_chunks[filtered_point_chunk_nr][prev_point_name] = copy.deepcopy(new_SIGRAG_chunks)
                    input_SIGRAG_chunks[filtered_point_chunk_nr]['recording dates'] = recording_chunk_date   
                    # input_chunk_list[filtered_point_chunk_nr].append(filtered_point_chunk)
            
            print(f"Starting agent chain for the points {prev_point_names} with the note {prev_note['noteComment']}")
            break_key = False
            input_chunk_slices = []
            
            # for input_chunk_slice_nr, input_chunk_slice in enumerate(input_chunk_list):
            
            for input_chunk_slice_nr, input_chunk_slice in range(0, len(input_SIGRAG_chunks), step_size):
                
                # input_chunk_slice_nr += 1
                
                # input_chunk_slices.append(input_chunk_slice)
                
                # if any(input_chunk_slice):
                recording_date_to_write = datetime.strptime(input_chunk_slice[-1]['chunk date'][0:19], '%Y-%m-%dT%H:%M:%S')
                note_date_to_write = datetime.strptime(prev_note['noteDateUTC'][0:19], '%Y-%m-%dT%H:%M:%S')
                days_between_note_and_recording = (recording_date_to_write - note_date_to_write).days
                
                
                agent_input_slices = input_SIGRAG_chunks[input_chunk_slice:input_chunk_slice+step_size]
                if not input_chunk_slice_nr:
                    input_string = f"""Your job is to make fault diagnosis predictions based on data from a condition monitoring (CM) database.
                    You are currently analysing signals from the asset at machine path {prev_asset_path}, with point names
                    and sensor types {input_dict}, where the positions of the point names and sensor types correspond to the positions of the
                    incoming recording slices.
                    The recording slices will feature one set of recordings per point name and sensor type, and iterate forward in time based on your requests.
                    Each recording features a set of associated SIGRAG chunks and a recording date.
                    The SIGRAG chunks are obtained by finding the most similar recordings from the CM database.
                    These SIGRAG chunks have associated annotations that describe fault properties of these chunks, time deltas which describe the time
                    distance between the SIGRAG chunk and its annotation, and asset and point information describing what machine you're analysing.
                    
                    If you are not ready to make a prediction, reply with "CONTINUE" and the user will provide another set of recording slices.
                    If you ask for more slices than are available, the user will break the loop and you may proceed with your analysis.
                    
                    Finally, respond with your analysis with the "reply_with_precition" tool.
                    
                    The first input and SIGRAG chunks are: {agent_input_slices}
                    """
                    prediction_agent_input = input_string
                    call_agent_flag = True
                else:
                    prediction_agent_input = f"""The next slices are {agent_input_slices}"""
                    call_agent_flag = True 
                # elif not input_chunk_slice_nr % 40:
                #     agent_input_slices = copy.deepcopy(input_chunk_slices)
                #     prediction_agent_input = f"""The next slices are {agent_input_slices}"""
                #     call_agent_flag = True
                #     input_chunk_slices = []
                # else:
                    
                #     call_agent_flag = False
                if call_agent_flag:
                    async for output in prediction_agent.astream({
                        "input": prediction_agent_input,
                        "chat_history": [],
                        }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
                        # stream_mode="updates" yields dictionaries with output keyed by node name
                        for key, value in output.items():
                            
                            if key == "exit_graph":
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                do_nothing = True
                            elif key == "prediction_thinker":
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                do_nothing = True
                            elif key == "get_signal":
                                break_key = True
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                break
                            elif key == "reply_with_prediction":
                                break_key = True
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                break
                            else:
                                print(f"Wrong tool {key} called in this part of the loop.")
                                break

                    input_chunk_slices = []
                            
                        # print("\n---\n")
                        
                if break_key:
                    break
                # if days_between_note_and_recording<0:
                #     break
            stupid_AI_counter = 0
            while not break_key:
                print(f"Something went wrong. {input_chunk_slice_nr} slices analysed and {days_between_note_and_recording} days until note.")
                async for output in prediction_agent.astream({
                    "input": """You have run out of chunks. You MUST make a prediction with what you have using "reply_with_prediction" or you will not progress. You will not
                    get any more chunks. Proceed through "reply_with_prediction", and if you have not detected any faults, please explain why.""",
                    "chat_history": [],
                    }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
                    for key, value in output.items():
                        print(f"Output from node '{key}':")
                        print("---")
                        print(value["intermediate_steps"][-1])
                if 'fault_diagnosis_prediction' in value['intermediate_steps'][0].tool_input:
                    break_key = True
                    break
                else:
                    time.sleep(10)
                    stupid_AI_counter+=1
                if stupid_AI_counter>5:
                    break
                    
            
            agent_prediction = value['intermediate_steps'][0].tool_input['fault_diagnosis_prediction']
            
            # input_chunk_ids = []
            # input_recordings = []
            # for input_chunk_element in input_chunk_slice:
            #     input_chunk_ids.append(input_chunk_element[0])
            #     input_recording = SIGRAG.get_recording_from_chunk_key(vector_store, input_chunk_element[0])
            #     if any(input_recording):
            #         input_recordings.append(input_recording[0])
            #     else:
            #         print(f"Error! No recording at {input_chunk_element[0]}")

            # replied_with_prediction = False
            
            # if using_recordings:
            #     prediction_input = f"""Ok. The most recent input slice is {input_recordings}, ordered the same as the SIGRAG chunks per the information in
            #             {input_dict}."""
            #     async for output in prediction_agent.astream({
            #         "input": prediction_input,
            #         "chat_history": [],
            #         }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #         # stream_mode="updates" yields dictionaries with output keyed by node name
            #         for key, value in output.items():
            #             if key == "reply_with_prediction":
            #                 replied_with_prediction = True
            #             else:
            #                 prediction_input = f"""You must reply with your prediction before proceeding."""
            #             #print(f"Output from node '{key}':")
            #             #print("---")
            #             #print(value["intermediate_steps"][-1])
            #         print("\n---\n")

            

            # async for output in prediction_agent.astream({
            #     "input": f"""Based on the SIGRAG chunks and recordings, what is your fault assessment?""",
            #     "chat_history": [],
            #     }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #     # stream_mode="updates" yields dictionaries with output keyed by node name
            #     #for key, value in output.items():
            #         #if key == "reply_with_prediction":
            #         #   replied_with_prediction = True
            #         #else:
            #         #    prediction_input = 
            #         #print(f"Output from node '{key}':")
            #         #print("---")
            #         #print(value["intermediate_steps"][-1])
            #     print("\n---\n")
            
            written_flag = False
            written_counter = 0
            
            #while not written_flag:
            
            async for output in evaluation_agent.astream({
                "input": f"""The true note is {prev_note['noteComment']} at a time delta of {days_between_note_and_recording} days. The prediction made was:
                {agent_prediction}. Was the analysis correct? Write it to file with the "write_prediction" tool.
                Remember to write the asset path {prev_asset_path} with point name and sensor types {input_dict} to file.""",
                "chat_history": [],
                }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
                # stream_mode="updates" yields dictionaries with output keyed by node name
                for key, value in output.items():
                    if key == "write_prediction":
                        written_flag=True
                        print("Writing prediction to file!")
                    else:
                        written_counter+=1
                        if written_counter>5:
                            written_flag=True
                    print(f"Output from node '{key}':")
                    print("---")
                    print(value["intermediate_steps"][-1])
                print("\n---\n")
            
            filtered_asset_chunks = [] 
            filtered_asset_chunks.append(filtered_input_chunks_input)
            point_types = []
            point_types.append(filtered_pointtype)
            prev_asset_path = filtered_path
            prev_point_names = []
            prev_point_names.append(filtered_pointname)
            prev_idnote = filtered_idnote
            prev_note = filtered_note
            made_pred_and_eval = True
            have_analysed_counter +=1
        else:
            filtered_asset_chunks = [] 
        #     filtered_asset_chunks.append(filtered_input_chunks_input)
            point_types = []
            point_types.append(filtered_pointtype)
            prev_asset_path = filtered_path
            prev_point_names = []
            prev_point_names.append(filtered_pointname)
            prev_idnote = filtered_idnote
            prev_note = filtered_note
            made_pred_and_eval = False
        
            
async def test_SIGRAG_agent_final(source_vector_store, filtered_vector_store, prediction_agent = None, evaluation_agent = None, using_recordings = False, step_size = 1, note_labels = False):
    if not prediction_agent:
        prediction_agent = SIGRAG_test.set_up_prediction_agent()
    if not evaluation_agent:
        evaluation_agent = SIGRAG_test.set_up_evaluation_agent()

    made_pred_and_eval = False
    filtered_asset_chunks = []
    prev_asset_path = ""
    prev_point_names = []
    point_types = []
    prev_note = []
    have_analysed_counter = 0
    #Image(prediction_agent.get_graph().draw_png())
    for doc_nr, (doc_id, doc) in enumerate(source_vector_store.items()):
        prev_note_flag = False
        doc_chunks = doc['recording_chunks']
        filtered_chunk_keys = list(doc_chunks.keys())[0:10]
        filtered_asset = doc['asset']
        filtered_path = " ".join(filtered_asset['Path'].split("\\")[2:])
        filtered_points = doc['point']
        filtered_pointtype = filtered_points['DetectionName']
        filtered_pointname = filtered_points['Name']
        filtered_note = doc['note']
        filtered_notecomment = filtered_note['noteComment']
        filtered_idnote = filtered_note['idNote']
        if not doc_nr:
            prev_idnote = filtered_idnote
        # if any(filtered_chunk_keys):
   
        print(filtered_notecomment)
        print(filtered_pointname)
        print(filtered_path)
        filtered_chunks = SIGRAG.get_chunks_from_chunk_keys(source_vector_store, filtered_chunk_keys)
        chunk_doc_id, _ = SIGRAG.get_keys_from_chunk_keys(source_vector_store, filtered_chunk_keys[0])
        filtered_emb_scores, filtered_asset_point_labels, filtered_asset_notes_labels = SIGRAG.compute_embedding_distance_final(
            filtered_vector_store, filtered_chunk_keys, filtered_chunks, chunk_doc_id, exclude_same_point = True)
        # filtered_input_chunk_key = list(filtered_emb_scores.keys())[0]
        filtered_top_results_with_notes = SIGRAG.top_results_short(filtered_vector_store, filtered_emb_scores, top_k = 5)
        # filtered_agent_input_asset, filtered_agent_input_point, filtered_agent_input_note = SIGRAG.get_hierarchy_from_chunk_key(vector_store, filtered_input_chunk_key)
        
        # filtered_input_path = " ".join(filtered_agent_input_asset['Path'].split("\\")[2:])
        
        # filtered_input_name = filtered_agent_input_point['Name']
        # filtered_input_date = filtered_agent_input_note['noteDateUTC']
        # filtered_input_note = filtered_agent_input_note['noteComment']
        
        # filtered_input_chunks = list(reversed(list(filtered_top_results_with_notes.items())))
        filtered_input_chunks_input = list(reversed(filtered_top_results_with_notes))
        # return filtered_input_chunks
        
        if prev_idnote == filtered_idnote and doc_nr != (len(source_vector_store)-1):
            filtered_asset_chunks.append(copy.deepcopy(filtered_input_chunks_input))
            point_types.append(filtered_pointtype)
            prev_asset_path = filtered_path
            prev_point_names.append(filtered_pointname)
            prev_note = filtered_note
        else:
            print(len(filtered_asset_chunks))
            input_dict = [{} for i in range(len(filtered_asset_chunks))]
            input_chunk_dict = {}
            max_rec_len = 0
            for filtered_point_chunks in filtered_asset_chunks:
                rec_len = len(filtered_point_chunks)
                if rec_len > max_rec_len:
                    max_rec_len = rec_len
            # input_chunk_list = [[] for i in range(max_rec_len)]
            input_SIGRAG_chunks = [{} for i in range(max_rec_len)]
            for point_nr, (filtered_point_chunks, point_type, prev_point_name) in enumerate(zip(filtered_asset_chunks, point_types, prev_point_names)):
                input_dict[point_nr]['sensor type'] = point_type
                input_dict[point_nr]['point name'] = prev_point_name
                # input_dict[point_nr]['SIGRAG chunks'] = filtered_point_chunks
                # input_SIGRAG_chunks[prev_point_name] = 
                print(len(filtered_point_chunks))
                for filtered_point_chunk_nr, filtered_point_chunk in enumerate(filtered_point_chunks):
                    # for SIGRAG_chunks in filtered_point_chunk['SIGRAG chunks']:
                    #     for SIGRAG_content in SIGRAG_chunks:
                    #         SIGRAG_content_keys = list(SIGRAG_content.keys())
                    #         SIGRAG_content_values = list(SIGRAG_content.values())
                            
                    #         for SIGRAG_content_key, SIGRAG_content_value in zip(SIGRAG_content_keys, SIGRAG_content_values):
                    #             if SIGRAG_content_key not in input_SIGRAG_chunks[prev_point_name].keys():
                    #                 input_SIGRAG_chunks[prev_point_name][SIGRAG_content_key] = []
                    #             input_SIGRAG_chunks[prev_point_name][SIGRAG_content_key].append(SIGRAG_content_value)
                    new_SIGRAG_chunks = {}
                    recording_chunk_date = filtered_point_chunk['chunk date']
                    for SIGRAG_chunk in filtered_point_chunk['SIGRAG chunks']:
                        for SIGRAG_content in SIGRAG_chunk:
                            #print(list(SIGRAG_content.items()))
                            SIGRAG_content_keys = list(SIGRAG_content.keys())
                            SIGRAG_content_values = list(SIGRAG_content.values())
                            
                            for SIGRAG_content_key, SIGRAG_content_value in zip(SIGRAG_content_keys, SIGRAG_content_values):
                                # print(SIGRAG_content_key)
                                # if SIGRAG_content_key == 'note content' or SIGRAG_content_key == 'note date':
                                    if SIGRAG_content_key not in new_SIGRAG_chunks:
                                        new_SIGRAG_chunks[SIGRAG_content_key] = []
                                    new_SIGRAG_chunks[SIGRAG_content_key].append(SIGRAG_content_value)
                    input_SIGRAG_chunks[filtered_point_chunk_nr]['recording dates'] = recording_chunk_date
                    input_SIGRAG_chunks[filtered_point_chunk_nr][prev_point_name] = copy.deepcopy(new_SIGRAG_chunks)
                    
                    # input_chunk_list[filtered_point_chunk_nr].append(filtered_point_chunk)
            input_string = f"""Your job is to make fault diagnosis predictions based on data from a condition monitoring (CM) database.
            You are currently analysing signals from the asset at machine path {prev_asset_path}, with point names
            and sensor types {input_dict}, where the positions of the point names and sensor types correspond to the positions of the
            incoming recording slices.
            The recording slices will feature one set of recordings per point name and sensor type, and iterate forward in time based on your requests.
            Each recording features a set of associated SIGRAG chunks and a recording date.
            The SIGRAG chunks are obtained by finding the most similar recordings from the CM database.
            These SIGRAG chunks have associated annotations that describe fault properties of these chunks, time deltas which describe the time
            distance between the SIGRAG chunk and its annotation, and asset and point information describing what machine you're analysing.
            
            If you are not ready to make a prediction, reply with "CONTINUE" and the user will provide another set of recording slices.
            If you ask for more slices than are available, the user will break the loop and you may proceed with your analysis.
            
            Finally, respond with your analysis with the "reply_with_precition" tool.
            
            The first input and SIGRAG chunks are: {input_SIGRAG_chunks[0:step_size]}
            """
            print(f"Starting agent chain for the points {prev_point_names} with the note {prev_note['noteComment']}")
            break_key = False
            # input_chunk_slices = []
            
            # for input_chunk_slice_nr, input_chunk_slice in enumerate(input_chunk_list):
            print(len(input_SIGRAG_chunks))
            for input_chunk_slice_nr, input_chunk_slice in enumerate(range(0, len(input_SIGRAG_chunks), step_size)):
                
                # input_chunk_slice_nr += 1
                
                # input_chunk_slices.append(input_chunk_slice)
                
                agent_input_slices = input_SIGRAG_chunks[input_chunk_slice:input_chunk_slice+step_size]
                recording_date_to_write = datetime.strptime(agent_input_slices[-1]['recording dates'][0:19], '%Y-%m-%dT%H:%M:%S')
                note_date_to_write = datetime.strptime(prev_note['noteDateUTC'][0:19], '%Y-%m-%dT%H:%M:%S')
                days_between_note_and_recording = (recording_date_to_write - note_date_to_write).days
                
                if not input_chunk_slice_nr:
                    prediction_agent_input = input_string
                    call_agent_flag = True
                else:
                    prediction_agent_input = f"""The next slices are {agent_input_slices}"""
                    call_agent_flag = True 
                # elif not input_chunk_slice_nr % 40:
                #     agent_input_slices = copy.deepcopy(input_chunk_slices)
                #     prediction_agent_input = f"""The next slices are {agent_input_slices}"""
                #     call_agent_flag = True
                #     input_chunk_slices = []
                # else:
                    
                #     call_agent_flag = False
                if call_agent_flag:
                    async for output in prediction_agent.astream({
                        "input": prediction_agent_input,
                        "chat_history": [],
                        }
                        , config = {"configurable": {"thread_id": str(doc_nr)}}
                        , stream_mode="updates"):
                        # stream_mode="updates" yields dictionaries with output keyed by node name
                        for key, value in output.items():
                            
                            if key == "exit_graph":
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                do_nothing = True
                            elif key == "prediction_thinker":
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                do_nothing = True
                            elif key == "get_signal":
                                break_key = True
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                break
                            elif key == "reply_with_prediction":
                                break_key = True
                                print(f"Output from node '{key}':")
                                print("---")
                                print(value["intermediate_steps"][-1])
                                break
                            else:
                                print(f"Wrong tool {key} called in this part of the loop.")
                                break

                            
                        # print("\n---\n")
                    
                if break_key:
                    break
                # if days_between_note_and_recording<0:
                #     break
            stupid_AI_counter = 0
            while not break_key:
                print(f"Something went wrong. {input_chunk_slice_nr} slices analysed and {days_between_note_and_recording} days until note.")
                async for output in prediction_agent.astream({
                    "input": """You have run out of chunks. You MUST make a prediction with what you have using "reply_with_prediction" or you will not progress. You will not
                    get any more chunks. Proceed through "reply_with_prediction", and if you have not detected any faults, please explain why.""",
                    "chat_history": [],
                    }
                    , config = {"configurable": {"thread_id": str(doc_nr)}}
                    , stream_mode="updates"):
                    for key, value in output.items():
                        print(f"Output from node '{key}':")
                        print("---")
                        print(value["intermediate_steps"][-1])
                if 'fault_diagnosis_prediction' in value['intermediate_steps'][0].tool_input:
                    break_key = True
                    break
                else:
                    time.sleep(10)
                    stupid_AI_counter+=1
                if stupid_AI_counter>5:
                    break
                    
            
            agent_prediction = value['intermediate_steps'][0].tool_input['fault_diagnosis_prediction']
            
            # input_chunk_ids = []
            # input_recordings = []
            # for input_chunk_element in input_chunk_slice:
            #     input_chunk_ids.append(input_chunk_element[0])
            #     input_recording = SIGRAG.get_recording_from_chunk_key(vector_store, input_chunk_element[0])
            #     if any(input_recording):
            #         input_recordings.append(input_recording[0])
            #     else:
            #         print(f"Error! No recording at {input_chunk_element[0]}")

            # replied_with_prediction = False
            
            # if using_recordings:
            #     prediction_input = f"""Ok. The most recent input slice is {input_recordings}, ordered the same as the SIGRAG chunks per the information in
            #             {input_dict}."""
            #     async for output in prediction_agent.astream({
            #         "input": prediction_input,
            #         "chat_history": [],
            #         }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #         # stream_mode="updates" yields dictionaries with output keyed by node name
            #         for key, value in output.items():
            #             if key == "reply_with_prediction":
            #                 replied_with_prediction = True
            #             else:
            #                 prediction_input = f"""You must reply with your prediction before proceeding."""
            #             #print(f"Output from node '{key}':")
            #             #print("---")
            #             #print(value["intermediate_steps"][-1])
            #         print("\n---\n")

            

            # async for output in prediction_agent.astream({
            #     "input": f"""Based on the SIGRAG chunks and recordings, what is your fault assessment?""",
            #     "chat_history": [],
            #     }, config = {"configurable": {"thread_id": str(doc_nr)}}, stream_mode="updates"):
            #     # stream_mode="updates" yields dictionaries with output keyed by node name
            #     #for key, value in output.items():
            #         #if key == "reply_with_prediction":
            #         #   replied_with_prediction = True
            #         #else:
            #         #    prediction_input = 
            #         #print(f"Output from node '{key}':")
            #         #print("---")
            #         #print(value["intermediate_steps"][-1])
            #     print("\n---\n")
            
            written_flag = False
            written_counter = 0
            
            #while not written_flag:
            
            async for output in evaluation_agent.astream({
                "input": f"""The true note is {prev_note['noteComment']} at a time delta of {days_between_note_and_recording} days. The prediction made was:
                {agent_prediction}. Was the analysis correct? Write it to file with the "write_prediction" tool.
                Remember to write the asset path {prev_asset_path} with point name and sensor types {input_dict} to file.""",
                "chat_history": [],
                }
                    , config = {"configurable": {"thread_id": str(doc_nr)}}
                    , stream_mode="updates"):
                # stream_mode="updates" yields dictionaries with output keyed by node name
                for key, value in output.items():
                    if key == "write_prediction":
                        written_flag=True
                        print("Writing prediction to file!")
                    else:
                        written_counter+=1
                        if written_counter>5:
                            written_flag=True
                    print(f"Output from node '{key}':")
                    print("---")
                    print(value["intermediate_steps"][-1])
                print("\n---\n")
            
            time.sleep(10)
            filtered_asset_chunks = [] 
            filtered_asset_chunks.append(filtered_input_chunks_input)
            point_types = []
            point_types.append(filtered_pointtype)
            prev_asset_path = filtered_path
            prev_point_names = []
            prev_point_names.append(filtered_pointname)
            prev_idnote = filtered_idnote
            prev_note = filtered_note
            made_pred_and_eval = True
            have_analysed_counter +=1
        
