#pragma once

//external libs
class Dataset {};
class Individual {};
#if !TEST
class Population {};
class Pairs {};
class CDTreeNode {};
class CTrainObjsInNode {};
class IDataSet {};
class DataSet {};
class CDataSetRT {};
class CCudaTimeStats {};

enum eModelType { NONE_MODEL, OPTIMAL_LEAF, OPTIMAL_LEAF_SLR, OPTIMAL_LEAF_SLR_FINAL, RANDOM_LEAF_SLR, SELECTED_LEAF_SLR, MIX_LEAF_SLR, FULL_LEAF_MLR, OPTIMAL_LEAF_MLR, M5_LEAF_MLR, M5_LEAF_MLR_FINAL, RANDOM_LEAF_MLR, INCREASE_LEAF_MLR, DECREASE_LEAF_MLR, CLEAN_LEAF_MLR };
#endif
