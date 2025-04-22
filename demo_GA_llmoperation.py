#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按照autogrow4参数设置(更改之前的理解错误)

demo_GAoperation.py - 分子进化与生成流程整合脚本

完整流程: 
1. 读取当前种群
2. 分子分解(decompose)
3. GPT生成新分子    
4. 种群融合与交叉
5. 再次分子分解
6. GPT再次生成新分子
7. 再次种群融合
8. 变异
9. 过滤
10. 分子对接

处理流程说明:
- Generation_0: 
- Generation_1到Generation_N: 
"""
#传参/参数  
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ["1","2","3"]  
import __future__

import argparse
import copy
import datetime
########################
# Run GA_llmoperations #
########################
PARSER = argparse.ArgumentParser()

# Allows the run commands to be submitted via a .json file.
PARSER.add_argument(
    "--json",
    "-j",
    metavar="param.json",
    help="Name of a json file containing all parameters. \
    Overrides other arguments.",
)
# Allows the run in debug mode. Doesn't delete temp files.
PARSER.add_argument(
    "--debug_mode",
    "-d",
    action="store_true",
    default=False,
    help="Run GA_llm in Debug mode. This keeps all \
    temporary files and adds extra print statements.",
)

# receptor information
PARSER.add_argument(
    "--filename_of_receptor",
    "-r",
    metavar="receptor.pdb",
    default="/data1/tgy/GA_llm/tutorial/PARP/4r6eA_PARP1_prepared.pdb",
    help="The path to the receptor file. Should be .pdb file.",
)
PARSER.add_argument(
    "--center_x",
    "-x",
    type=float,
    help="x-coordinate for the center of the pocket to be tested by docking. (Angstrom)",
)
PARSER.add_argument(
    "--center_y",
    "-y",
    type=float,
    help="y-coordinate for the center of the pocket to be tested by docking. (Angstrom)",
)
PARSER.add_argument(
    "--center_z",
    "-z",
    type=float,
    help="z-coordinate for the center of the pocket to be tested by docking. (Angstrom)",
)

PARSER.add_argument(
    "--size_x",
    type=float,
    help="dimension of box to dock into in the x-axis (Angstrom)",
)
PARSER.add_argument(
    "--size_y",
    type=float,
    help="dimension of box to dock into in the y-axis (Angstrom)",
)
PARSER.add_argument(
    "--size_z",
    type=float,
    help="dimension of box to dock into in the z-axis (Angstrom)",
)


# Input/Output directories
PARSER.add_argument(
    "--root_output_folder",
    "-o",
    type=str,
    default="/data1/tgy/GA_llm/tutorial/PARP/output_demoGA_llm",
    help="The Path to the folder which all output files will be placed.",
)
#初始种群！记得仔细看下这里！
PARSER.add_argument(
    "--source_compound_file",
    "-s",
    type=str,
    default="/data1/tgy/GA_llm/tutorial/PARP/PARP_source_compounds.smi",
    help="PATH to the file containing the source compounds. It must be \
    tab-delineated .smi file. These ligands will seed the first generation.",
)
PARSER.add_argument(
    "--filter_source_compounds",
    choices=[True, False, "True", "False", "true", "false"],
    default=True,
    help="If True source ligands from source_compound_file will be \
    filter using the user defined filter choices prior to the 1st generation being \
    created. If False, ligands which would fail the ligand filters could seed \
    the 1st generation. Default is True.",
)
PARSER.add_argument(
    "--use_docked_source_compounds",
    choices=[True, False, "True", "False", "true", "false"],
    default=False,
    help="If True   source ligands will be docked prior to seeding generation 1. \
    If True and the source_compound file already has docking/fitness metric score \
    in -2 column of .smi file, it will not redock but reuse the scores from \
    the source_compound_file.\
    If True and no fitness metric score in -2 column of .smi file, it will \
    dock each ligand from the source_compound_file and displayed as generation 0.\
    If False, generation 1 will be randomly seeded by the source compounds with \
    no preference and there will be no generation 0. \
    If performing multiple simulations using same source compounds and protein, \
    we recommend running once this and using the generation 0 ranked file as the \
    source_compound_file for future simulations. \
    Default is True.",
)
PARSER.add_argument(
    "--start_a_new_run",
    action="store_true",
    default=False,
    help="If False make a new folder and start a fresh simulation with Generation 0.  \
    If True find the last generation in the root_output_folder and continue to fill.\
    Default is False.",
)
#蒙特卡洛搜索
# SmilesMerge Settings
PARSER.add_argument(
    "--max_time_MCS_prescreen",
    type=int,
    default=1,
    help="amount time the pre-screen MCS times out. Time out doesnt prevent \
    mcs matching just takes what it has up to that point",
)
PARSER.add_argument(
    "--max_time_MCS_thorough",
    type=int,
    default=1,
    help="amount time the thorough MCS times out. Time out doesnt prevent \
    mcs matching just takes what it has up to that point",
)
PARSER.add_argument(
    "--min_atom_match_MCS",
    type=int,
    default=4,
    help="Determines the minimum number of atoms in common for a substructurematch. \
    The higher the more restrictive, but the more likely for two ligands not to match",
)
PARSER.add_argument(
    "--protanate_step",
    action="store_true",#在使用的时候指定：true
    default=False,
    help="Indicates if Smilesmerge uses protanated mols (if true) or deprot \
    (if False) SmilesMerge is 10x faster when deprotanated",
)
# Mutation Settings
PARSER.add_argument(
    "--rxn_library",#反应库
    choices=["click_chem_rxns", "robust_rxns", "all_rxns", "Custom"],
    default="all_rxns",
    help="This set of reactions to be used in Mutation. \
    If Custom, one must also provide rxn_file Path and function_group_library path",
)
PARSER.add_argument(
    "--rxn_library_file",#反应库地址
    type=str,
    default="/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_rxn_library.json",
    help="This PATH to a Custom json file of SMARTS reactions to use for Mutation. \
    Only provide if using the Custom option for rxn_library.",
)
PARSER.add_argument(
    "--function_group_library",#官能团地址
    type=str,
    default="/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/All_Rxns_functional_groups.json",
    help="This PATH for a dictionary of functional groups to be used for Mutation. \
    Only provide if using the Custom option for rxn_library.",
)
PARSER.add_argument(
    "--complementary_mol_directory",#互补分子目录
    type=str,
    default="/data1/tgy/GA_llm/autogrow/operators/mutation/smiles_click_chem/reaction_libraries/all_rxns/complementary_mol_dir",
    help="This PATH to the directory containing all the molecules being used \
    to react with. The directory should contain .smi files contain SMILES of \
    molecules containing the functional group represented by that file. Each file \
    should be named with the same title as the functional groups described in \
    rxn_library_file & function_group_library +.smi \
    All Functional groups specified function_group_library must have its \
    own .smi file. We recommend you filter these dictionaries prior to GA_llm \
    for the Drug-likeliness and size filters you will Run GA_llm with.",
)


# processors and multithread mode
PARSER.add_argument(
    "--number_of_processors",
    "-p",
    type=int,
    metavar="N",
    default=-1,#默认使用所有可得的CPU
    help="Number of processors to use for parallel calculations. Set to -1 for all available CPUs.",
)
PARSER.add_argument(
    "--multithread_mode",    
    choices=["mpi", "multithreading", "serial"],
    default="multithreading",#默认使用多线程
    help="Determine what style \
    multithreading: mpi, multithreading, or serial. serial will override \
    number_of_processors and force it to be on a single processor.",
)

# Genetic Algorithm Options
PARSER.add_argument(
    "--selector_choice",
    choices=["Roulette_Selector", "Rank_Selector", "Tournament_Selector"],
    default="Roulette_Selector",#默认使用轮盘赌选择器
    help="This determines whether the fitness criteria are chosen by a Weighted Roulette, \
    Ranked, or Tournament style Selector. The Rank option is a non-redundant selector.\
    Roulette and Tournament chose without replacement and are stoichastic options. \
    Warning do not use Rank_Selector for small runs as there is potential that \
    the number of desired ligands exceed the number of ligands to chose from.",
)
PARSER.add_argument(
    "--tourn_size",#锦标赛选择器大小---对应上面第三种选择
    type=float,
    default=0.1,
    help="If using the Tournament_Selector this determines the size of each \
    tournament. The number of ligands used for each tournament will the \
    tourn_size * the number of considered ligands.",
)

# Seeding next gen and diversity
#这些参数共同决定了算法在"开发"（利用已知优良解）和"探索"（寻找新区域）之间的平衡策略。
PARSER.add_argument(
    "--top_mols_to_seed_next_generation_first_generation",#“fitness:docking_score"
    type=int,
    help="Number of mols that seed next generation, for the first generation.\
    Should be less than number_of_crossovers_first_generation + number_of_mutations_first_generation\
    If not defined it will default to top_mols_to_seed_next_generation",
)
PARSER.add_argument(
    "--top_mols_to_seed_next_generation",
    type=int,
    default=10,
    help="Number of mols that seed next generation, for all generations after the first.\
    Should be less than number_of_crossovers_first_generation \
    + number_of_mutations_first_generation",
)
PARSER.add_argument(
    "--diversity_mols_to_seed_first_generation",#分子指纹(fingerprints)计算分子间的结构差异性，选择结构差异大的分子
    type=int,
    default=10,
    help="Should be less than number_of_crossovers_first_generation \
    + number_of_mutations_first_generation",
)
PARSER.add_argument(  #随着代数增加，算法会减少对多样性的关注，增加对已知优良解的利用
    "--diversity_seed_depreciation_per_gen",
    #动态控制实现：每代多样性种子数量 = diversity_mols_to_seed_first_generation - (当前代数 × diversity_seed_depreciation_per_gen)
    type=int,
    default=2,
    help="Each gen diversity_mols_to_seed_first_generation will decrease this amount",
)

# Populations settings
PARSER.add_argument(
    "--num_generations",
    type=int,
    default=10,
    help="The number of generations to be created.",
)
PARSER.add_argument(
    "--number_of_crossovers_first_generation",
    type=int,
    help="The number of ligands which will be created via crossovers in the \
    first generation. If not defined it will default to number_of_crossovers",
)
PARSER.add_argument(
    "--number_of_mutants_first_generation",
    type=int,
    help="The number of ligands which will be created via mutation in \
    the first generation. If not defined it will default to number_of_mutants",
)
PARSER.add_argument(
    "--number_elitism_advance_from_previous_gen_first_generation",
    type=int,
    help="The number of ligands chosen for elitism for the first generation \
    These will advance from the previous generation directly into the next \
    generation.  This is purely advancing based on Docking/Rescore fitness. \
    This does not select for diversity. If not defined it will default to \
    number_elitism_advance_from_previous_gen",
)
PARSER.add_argument(
    "--number_of_crossovers",
    type=int,
    default=10,
    help="The number of ligands which will be created via crossover in each \
    generation besides the first",
)
PARSER.add_argument(
    "--number_of_mutants",
    type=int,
    default=10,
    help="The number of ligands which will be created via mutation in each \
    generation besides the first.",
)
PARSER.add_argument(
    "--number_elitism_advance_from_previous_gen",
    type=int,
    default=10,
    help="The number of ligands chosen for elitism. These will advance from \
    the previous generation directly into the next generation. \
    This is purely advancing based on Docking/Rescore \
    fitness. This does not select for diversity.",
)
PARSER.add_argument(
    "--redock_elite_from_previous_gen",
    choices=[True, False, "True", "False", "true", "false"],
    default=False,
    help="If True than ligands chosen via Elitism (ie advanced from last generation) \
    will be passed through Gypsum and docked again. This provides a better exploration of conformer space \
    but also requires more computation time. If False, advancing ligands are simply carried forward by \
    copying the PDBQT files.",
)

####### FILTER VARIABLES
PARSER.add_argument(
    "--LipinskiStrictFilter",
    action="store_true",
    default=False,
    help="Lipinski filters for orally available drugs following Lipinski rule of fives. \
    Filters by molecular weight, logP and number of hydrogen bond donors and acceptors. \
    Strict implementation means a ligand must pass all requirements.",
)
PARSER.add_argument(
    "--LipinskiLenientFilter",
    action="store_true",
    default=False,
    help="Lipinski filters for orally available drugs following Lipinski rule of fives. \
    Filters by molecular weight, logP and number of hydrogen bond donors and acceptors. \
    Lenient implementation means a ligand may fail all but one requirement and still passes.",
)
PARSER.add_argument(
    "--GhoseFilter",
    action="store_true",
    default=False,
    help="Ghose filters for drug-likeliness; filters by molecular weight,\
    logP and number of atoms.",
)
PARSER.add_argument(
    "--GhoseModifiedFilter",
    action="store_true",
    default=False,
    help="Ghose filters for drug-likeliness; filters by molecular weight,\
    logP and number of atoms. This is the same as the GhoseFilter, but \
    the upper-bound of the molecular weight restrict is loosened from \
    480Da to 500Da. This is intended to be run with Lipinski Filter and \
    to match GA_llm 3's Ghose Filter.",
)
PARSER.add_argument(
    "--MozziconacciFilter",
    action="store_true",
    default=False,
    help="Mozziconacci filters for drug-likeliness; filters by the number of \
    rotatable bonds, rings, oxygens, and halogens.",
)
PARSER.add_argument(
    "--VandeWaterbeemdFilter",
    action="store_true",
    default=False,
    help="VandeWaterbeemd filters for drug likely to be blood brain barrier permeable. \
    Filters by the number of molecular weight and Polar Sureface Area (PSA).",
)
PARSER.add_argument(
    "--PAINSFilter",
    action="store_true",
    default=False,
    help="PAINS filters against Pan Assay Interference Compounds using \
    substructure a search.",
)
PARSER.add_argument(
    "--NIHFilter",
    action="store_true",
    default=False,
    help="NIH filters against molecules with undersirable functional groups \
    using substructure a search.",
)
PARSER.add_argument(
    "--BRENKFilter",
    action="store_true",
    default=False,
    help="BRENK filter for lead-likeliness, by matching common false positive \
    molecules to the current mol.",
)
PARSER.add_argument(
    "--No_Filters",
    action="store_true",
    default=False,
    help="No filters will be applied to compounds.",
)
PARSER.add_argument(
    "--alternative_filter",
    action="append",
    help="If you want to add Custom filters to the filter child classes \
    Must be a list of lists \
    [[name_filter1, Path/to/name_filter1.py],[name_filter2, Path/to/name_filter2.py]]",
)

# dependency variables
# DOCUMENT THE file conversion for docking inputs
#对接时 文件转换工具
PARSER.add_argument(
    "--conversion_choice",
    choices=["MGLToolsConversion", "ObabelConversion", "Custom"],
    default="MGLToolsConversion",
    help="Determines how .pdb files will be converted \
    to the final format for docking. For Autodock Vina and QuickVina style docking software, \
    files must be in .pdbqt format. MGLToolsConversion: uses MGLTools and is the \
    recommended converter. MGLTools conversion is required for NNScore1/2 rescoring. \
    ObabelConversion: uses commandline obabel. Easier to install but Vina docking has \
    been optimized with MGLTools conversion.",
)
PARSER.add_argument(
    "--custom_conversion_script",
    metavar="custom_conversion_script",
    default="",
    help="The path to a python script for which is used to convert \
    ligands. This is required for custom conversion_choice choices. \
    Must be a list of strings \
    [name_custom_conversion_class, Path/to/name_custom_conversion_class.py]",
)
PARSER.add_argument(
    "--mgltools_directory",
    metavar="mgltools_directory",
    help="Required if using MGLTools conversion option \
    (conversion_choice=MGLToolsConversion) \
    Path may look like: /home/user/MGLTools-1.5.6/",
)
PARSER.add_argument(
    "--mgl_python",
    metavar="mgl_python",
    required=False,
    default="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6/bin/pythonsh",
    help="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6/bin/pythonsh",
)
PARSER.add_argument(
    "--prepare_ligand4.py",
    metavar="prepare_ligand4.py",
    required=False,
    default="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
    help="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
)
PARSER.add_argument(
    "--prepare_receptor4.py",
    metavar="prepare_receptor4.py",
    required=False,
    help="/data1/ytg/GA_llm/mgltools_x86_64Linux2_1.5.6/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py",
)
PARSER.add_argument(
    "--obabel_path",
    help="required if using obabel conversion \
    option (conversion_choice=ObabelConversion).\
    Path may look like PATH/envs/py37/bin/obabel; \
    may be found on Linux by running: which obabel",
)

# docking
PARSER.add_argument(
    "--dock_choice",
    metavar="dock_choice",
    default="QuickVina2Docking",
    choices=["VinaDocking", "QuickVina2Docking", "Custom"],
    help="dock_choice assigns which docking software module to use.",
)
PARSER.add_argument(
    "--docking_executable",
    metavar="docking_executable",
    default="/data1/ytg/GA_llm/autodock_vina_1_1_2_linux_x86/bin/vina",
    help="path to the docking_executable",
)
PARSER.add_argument(
    "--docking_exhaustiveness",
    metavar="docking_exhaustiveness",
    default=None,
    help="exhaustiveness of the global search (roughly proportional to time. \
    see docking software for settings. Unless specified GA_llm uses the \
    docking softwares default setting. For AutoDock Vina 1.1.2 that is 8",
)
PARSER.add_argument(
    "--docking_num_modes",
    metavar="docking_num_modes",
    default=None,
    help=" maximum number of binding modes to generate in docking. \
    See docking software for settings. Unless specified GA_llm uses the \
    docking softwares default setting. For AutoDock Vina 1.1.2 that is 9",
)
PARSER.add_argument(
    "--docking_timeout_limit",
    type=float,
    default=120,
    help="The maximum amount of time allowed to dock a single ligand into a \
    pocket in seconds. Many factors influence the time required to dock, such as: \
    processor speed, the docking software, rotatable bonds, exhaustiveness docking,\
    and number of docking modes... \
    The default docking_timeout_limit is 120 seconds, which is excess for most \
    docking events using QuickVina2Docking under default settings. If run with \
    more exhaustive settings or with highly flexible ligands, consider increasing \
    docking_timeout_limit to accommodate. Default docking_timeout_limit is 120 seconds",
)
PARSER.add_argument(
    "--custom_docking_script",
    metavar="custom_docking_script",
    default="",
    help="The name and path to a python script for which is used to \
    dock ligands. This is required for Custom docking choices Must be a list of \
    strings [name_custom_conversion_class, Path/to/name_custom_conversion_class.py]",
)

# scoring
#打分
PARSER.add_argument(
    "--scoring_choice",
    metavar="scoring_choice",
    choices=["VINA", "NN1", "NN2", "Custom"],
    default="VINA",
    help="The scoring_choice to use to assess the ligands docking fitness. \
    Default is using Vina/QuickVina2 ligand affinity while NN1/NN2 use a Neural Network \
    to assess the docking pose. Custom requires providing a file path for a Custom \
    scoring function. If Custom scoring function, confirm it selects properly, \
    GA_llm is largely set to select for a more negative score.",
)
PARSER.add_argument(
    "--rescore_lig_efficiency",
    action="store_true",
    default=False,
    help="This will divide the final scoring_choice output by the number of \
    non-Hydrogen atoms in the ligand. This adjusted ligand efficiency score will \
    override the scoring_choice value. This is compatible with all scoring_choice options.",
)
PARSER.add_argument(
    "--custom_scoring_script",
    metavar="custom_scoring_script",
    type=str,
    default="",
    help="The path to a python script for which is used to \
    assess the ligands docking fitness. GA_llm is largely set to select for a most \
    negative scores (ie binding affinity the more negative is best). Must be a list of \
    strings [name_custom_conversion_class, Path/to/name_custom_conversion_class.py]",
)

# gypsum # max variance is the number of conformers made per ligand
PARSER.add_argument(
    "--max_variants_per_compound",
    type=int,
    default=3,
    help="number of conformers made per ligand. \
    See Gypsum-DL publication for details",
)
PARSER.add_argument(
    "--gypsum_thoroughness",
    "-t",
    type=str,
    help="How widely Gypsum-DL will search for \
    low-energy conformers. Larger values increase \
    run times but can produce better results. \
    See Gypsum-DL publication for details",
)
PARSER.add_argument(
    "--min_ph",
    metavar="MIN",
    type=float,
    default=6.4,
    help="Minimum pH to consider.See Gypsum-DL \
    and Dimorphite-D publication for details.",
)
PARSER.add_argument(
    "--max_ph",
    metavar="MAX",
    type=float,
    default=8.4,
    help="Maximum pH to consider.See Gypsum-DL \
    and Dimorphite-D publication for details.",
)
PARSER.add_argument(
    "--pka_precision",
    metavar="D",
    type=float,
    default=1.0,
    help="Size of pH substructure ranges. See Dimorphite-DL \
    publication for details.",
)
PARSER.add_argument(
    "--gypsum_timeout_limit",
    type=float,
    default=15,
    help="Maximum time gypsum is allowed to run for a given ligand in seconds. \
    On average Gypsum-DL takes on several seconds to run for a given ligand, but \
    factors such as mol size, rotatable bonds, processor speed, and gypsum \
    settings (ie gypsum_thoroughness or max_variants_per_compound) will change \
    how long it takes to run. If increasing gypsum settings it is best to increase \
    the gypsum_timeout_limit. Default gypsum_timeout_limit is 15 seconds",
)

# Reduce files down. This compiles and compresses the files in the PDBs folder
# (contains docking outputs, pdb, pdbqt...). This reduces the data size and
# makes data transfer quicker, but requires running the
# file_concatenation_and_compression.py in the Utility script folder to
# separate these files out for readability.
PARSER.add_argument(
    "--reduce_files_sizes",
    choices=[True, False, "True", "False", "true", "false"],
    default=True,
    help="Run this combines all files in the PDBs folder into a \
    single text file. Useful when data needs to be transferred.",
)

# Make a line plot of the simulation at the end of the run.
PARSER.add_argument(
    "--generate_plot",
    choices=[True, False, "True", "False", "true", "false"],
    default=True,
    help="Make a line plot of the simulation at the end of the run.",
)

# mpi mode pre-Run so there are python cache files without EOF Errors
PARSER.add_argument(
    "--cache_prerun",
    "-c",
    action="store_true",
    help="Run this before running gypsum in mpi-mode.",
)