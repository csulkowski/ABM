import re
import numpy as np
import os
import pickle

class ABM:
    """
    Object containing model preprocessing in addition to routines related to
    running the model and estimation.
    ...
    
    Parameters
    ----------
    modfile : string
        filename of the model file.
        
    cache : bool
        True will overwrite any previously generated compiled model.
        False will not overwrite, but will still preprocess the model.
        Note that if the compiledModel file does not exist, it will be 
        built regardless.
    
    Attributes
    ----------
    blocks : dict
        Contains parsed output of model preprocessing.
    
    params : ndarray
        Numpy array containing model parameters.
    
    supported_numpy_functions : list of str
        Includes all the numpy functions allowed during model compilation.
        Most functions can be freely added in principle, however, others may require
        additional parsing. Modify with care.
        If a variable in the model is named exactly the same as any function in this list,
        the preprocessor will throw an error.
        
    supported_numpy_arguments: list of str
        Includes all the numpy function arguments allowed during model compilation.
        If a variable in the model is named exactly the same as any variable in this list,
        the preprocessor will throw an error.
        
    supported_statements: list of str
        Supported statements in model file. 
        Warning! Any additional statement requires a proper implementation.
    
    hard_coded_names: list of str
        Variable names reserved for the final compiled model.
        Variables in the model cannot share a name with hard coded variables,
        however, the model file can refer to them in equations, endogenous variable 
        initialization, model steps etc.
        
    Methods
    ----------
    run(iterations=1000, plot=False, params = [], verbose = False)
        Runs the model for the specified amount of iterations.
        Returns dseries object containing simulated data.
        
    estimate(dbase = dbase, initial_params = params, estimated_vars = [], iterations = 1000, batch = 10, start = 0, step=0.001, parallel = False):
        Estimates model parameters using Metropolis-Hastings with simulated maximum likelihood. 
        Kukacka, J., & Barunik, J. (2017). Estimation of financial agent-based models with simulated maximum likelihood.
             Journal of Economic Dynamics and Control, 85, 21–45. doi:10.1016/j.jedc.2017.09.006 
    
    plot_normalize(normalization_input={})
        Sets variables to be normalized during plotting and on which variable they should be normalized.
        Example: {'OMEGAH':'Pm'}
            The variable OMEGAH will be normalized using the values of the variable Pm
            
    numba_functions()
        Class containing user made functions. Main use is to permit numba compilation with
        custom built versions of incompatible numpy functions. 
        Functions can also be specifically built for use in the model.
    
    """
    def __init__(self, modfile, cache=False):
        # Initialize variables necessary regardless of compilation
        self._plot_normalization = {}
        
        if cache == False or not(os.path.exists("compiledModel.py")):
            writeModel = True
        else:
            writeModel = False
            
        self.modfile = modfile
        block_list_arguments = ['SETTINGS', 'AGENTS', 'ENDO_VAR', 'ENDO_MAT', 'ENDO_INIT', 'EXO_PARAM', 'MAT_TYPE', 'ENDO_EQ', 'STEPS']
        default_agent_options = ['depend', 'num', 'tag', 'iterator', 'group']
        default_eq_options = ['equation', 'condition']
        default_settings_options = ['isclose', 'numba', 'show_values']
        # The list below contains the numpy functions which are supported.
        # Other functions may be added at the user's own risk
        self.supported_numpy_functions = ['log', 'exp', 'max', 'fmax', 'min', 'fmin', 'sum', 'mean', 'std', 'random.randint',
                                     'random.uniform', 'random.normal', 'random.permutation', 'ones', 'zeros', 'floor', 'ceil', 'prod',
                                     'arange', 'concatenate', 'nansum','isnan', 'isinf', 'round', 'size', 'shape', 'int_', 'nan', 'argsort',
                                     'reshape', 'operator.__lt__', 'operator.__gt__', 'logical_not',
                                     'dtype', 'int64', 'abs', 'inf', 'amin']

        self.supported_numpy_arguments = ['initial']

        #Alternatively, functions unsupported by Numba can be included in the following class
        self.numba_func = self.numba_functions()

        self.supported_statements = ['while', 'for', 'if', 'break', 'quit']

        self.hard_coded_names = ['iterations', 't']

        with open(self.modfile, 'r') as f:
            mod = f.readlines()

        # Remove comments, newlines, and whitespace from mod file
        mod = [re.sub('//.*|\\n| ', '', l) for l in mod]
        # Remove empty lines from mod file
        mod = [l for l in mod if (len(l) > 0 and not(l.isspace()))]

        # Get included blocks
        included_blocks = [b for b in mod if b in block_list_arguments]
        # Get order of blocks in mod
        block_order = [mod.index(b) for b in included_blocks]
        # Sort included blocks by appearance
        included_blocks = [x for y, x in sorted(zip(block_order, included_blocks))]
        # Get index of semi-colons
        semi_colons_mod = [l for l in range(len(mod)) if mod[l] == ';']

        # Sanity checks
        if 'AGENTS' not in included_blocks:
            raise ValueError('Agents need to be declared')
        if len(list(set(included_blocks))) != len(included_blocks):
            for b in list(set(included_blocks)):
                included_blocks.remove(b)
            for b in included_blocks:
                raise ValueError(b+' was already declared')
        if len(included_blocks) > len(semi_colons_mod):
            raise ValueError('Syntax Error: There are more declared blocks than semi-colons')
        if len(included_blocks) < len(semi_colons_mod):
            raise ValueError('Syntax Error: There are more semi-colons than declared blocks')

        # Get blocks
        self.blocks = {}
        for i, b in enumerate(included_blocks):
            self.blocks[b] = mod[mod.index(included_blocks[i])+1:semi_colons_mod[i]]

        # Order blocks back into order needed to compile
        included_blocks = [b for b in block_list_arguments if b in included_blocks]

        # Iterate through blocks to build modfile dictionary "self.blocks".
        # This dictionary will contain all the elements needed to compile the model.
        for b in included_blocks:
            if b == 'SETTINGS':
                unparsed_settings = self.blocks[b]
                parsed_settings = {}
                for option in default_settings_options:
                    if option == 'isclose':
                        parsed_settings['isclose'] = '0'
                    if option == 'numba':
                        parsed_settings['numba'] = 'False'
                    if option == 'show_values':
                        parsed_settings['show_values'] = 'False'
                for option in unparsed_settings:
                    option_split = option.split('=')
                    if len(option_split) != 2:
                        raise ValueError('Syntax Error: Malformed setting \"'+option+'\"')
                    if option_split[0] not in default_settings_options:
                        raise ValueError('Syntax Error: Setting \"'+option_split[0]+'\" is not recognized')
                    setting = option_split[0]
                    val = option_split[1]
                    parsed_settings[setting] = val
                #Check that values are valid for settings
                if parsed_settings['numba'] != 'True' and parsed_settings['numba']!= 'False':
                    raise ValueError('Syntax Error: Numba setting can only take True or False, current setting: '+parsed_settings['numba'])

                self.blocks[b] = parsed_settings

            elif b == 'AGENTS':
                unparsed_agents = self.blocks[b]
                parsed_agents = {}
                for agent in unparsed_agents:
                    agent_split = agent.split('=', 1)
                    name = agent_split[0]
                    agent_options = {}
                    for options in default_agent_options:
                        if options=='iterator' or options=='group':
                            agent_options[options] = []
                        else:
                            agent_options[options] = ''
                    options = [o.group() for o in re.finditer('\w+=(\w+[,|\]])+', agent_split[1])]
                    for opt in options:
                        option = opt.split('=')
                        option_val = option[1][:-1].split(',')
                        if option[0]!='iterator' and option[0]!='group':
                            agent_options[option[0]] = option_val[0]
                        else:
                            agent_options[option[0]] = option_val
                    parsed_agents[name] = agent_options

                self.blocks[b] = parsed_agents
                # Make list of all tags for following sanity checks
                self.model_A_TAGS = []
                for a in list(parsed_agents.keys()):
                    self.model_A_TAGS.append(parsed_agents[a]['tag'])

                # Make list of all iterators for following sanity checks
                self.model_A_ITERATOR = []
                for a in list(parsed_agents.keys()):
                    self.model_A_ITERATOR = self.model_A_ITERATOR + parsed_agents[a]['iterator']

                # Make list of all tags for following sanity checks
                self.model_A_GROUPS = []
                for a in list(parsed_agents.keys()):
                    self.model_A_GROUPS = self.model_A_GROUPS + parsed_agents[a]['group']

            elif b == 'ENDO_VAR':
                unparsed_ENDO_VAR = self.blocks[b]
                parsed_ENDO_VAR = {}
                agents = list(self.blocks['AGENTS'].keys())
                # Get agents in ENDO_VAR
                agent_ENDO_VAR = [a for a in unparsed_ENDO_VAR if a in agents]

                # Sanity check
                if len(agent_ENDO_VAR) != len(agents):
                    raise ValueError('Mismatch between number of agents declared and ENDO_VAR')
                if len(list(set(unparsed_ENDO_VAR))) != len(unparsed_ENDO_VAR):
                    for b in list(set(unparsed_ENDO_VAR)):
                        unparsed_ENDO_VAR.remove(b)
                    for b in unparsed_ENDO_VAR:
                        raise ValueError('Variable ' + b + ' is declared multiple times in ENDO_VAR')

                # Get order of agents in ENDO_VAR
                agent_order = [unparsed_ENDO_VAR.index(a) for a in agents]
                # Sort agents in ENDO_VAR by appearance
                agents = [x for y, x in sorted(zip(agent_order, agents))]

                for a in reversed(agents):
                    parsed_ENDO_VAR[a] = unparsed_ENDO_VAR[unparsed_ENDO_VAR.index(a)+1:]
                    unparsed_ENDO_VAR = unparsed_ENDO_VAR[:unparsed_ENDO_VAR.index(a)]

                # Make list of all vars for following sanity checks
                self.model_ENDO_VARS = []
                for a in list(parsed_ENDO_VAR.keys()):
                    self.model_ENDO_VARS = self.model_ENDO_VARS + parsed_ENDO_VAR[a]
                self.blocks[b] = parsed_ENDO_VAR

            elif b == 'ENDO_MAT':
                unparsed_ENDO_MAT = self.blocks[b]
                parsed_ENDO_MAT = {}

                for m in unparsed_ENDO_MAT:
                    m_orig = m
                    m = m.split('=')
                    if len(m) != 2:
                        raise ValueError('Syntax Error: '+str(m_orig))
                    reg_expression = '\+|-|\*|\/|@|\(|\)|\]|\[|:|<|>|=|!|\.T| |\\\\'
                    m_raw = re.sub(reg_expression, ',', m[1])
                    m_raw = m_raw.split(',')
                    m_raw = [term for term in m_raw if term != '']
                    m_tags = [term for term in m_raw if term in self.model_A_TAGS]
                    m_iterator = []
                    for m_tag in m_tags:
                        for a in list(self.blocks['AGENTS'].keys()):
                            if self.blocks['AGENTS'][a]['tag'] == m_tag:
                                m_iterator.append(self.blocks['AGENTS'][a]['iterator'][0])
                    m_dict = {}
                    m_dict['eq'] = m[1]
                    m_dict['tags'] = m_iterator
                    parsed_ENDO_MAT[m[0]] = m_dict

                # Sanity check
                if len(list(set(parsed_ENDO_MAT.keys()))) != len(parsed_ENDO_MAT.keys()):
                    for b in list(set(unparsed_ENDO_MAT)):
                        unparsed_ENDO_MAT.remove(b)
                    for b in unparsed_ENDO_MAT:
                        raise ValueError('Matrix ' + b + ' is declared multiple times in ENDO_MAT')

                # Make list of all vars for following sanity checks
                self.model_ENDO_MAT = list(parsed_ENDO_MAT.keys())

                self.blocks[b] = parsed_ENDO_MAT

            elif b == 'ENDO_INIT':
                unparsed_ENDO_INIT = self.blocks[b]
                parsed_ENDO_INIT = {}
                for i in unparsed_ENDO_INIT:
                    i_orig = i
                    i = i.split('=')
                    if len(i) != 2:
                        raise ValueError('Syntax Error: '+str(i_orig))
                    parsed_ENDO_INIT[i[0]] = i[1]
                self.blocks[b] = parsed_ENDO_INIT

            elif b == 'EXO_PARAM':
                unparsed_EXO_PARAM = self.blocks[b]
                parsed_EXO_PARAM = {}
                for p in unparsed_EXO_PARAM:
                    p_orig = p
                    p = p.split('=')
                    if len(p) != 2:
                        raise ValueError('Syntax Error: '+str(p_orig))
                    parsed_EXO_PARAM[p[0]] = p[1]

                # Make list of all exo param for following sanity checks
                self.model_EXO_PARAM = list(parsed_EXO_PARAM.keys())
                self.blocks[b] = parsed_EXO_PARAM

            elif b == 'MAT_TYPE':
                unparsed_MAT_TYPE = self.blocks[b]
                parsed_MAT_TYPE = {}
                for type in unparsed_MAT_TYPE:
                    type_orig = type
                    type = type.split('=')
                    if len(type) != 2:
                        raise ValueError('Syntax Error: '+str(type_orig))
                    parsed_MAT_TYPE[type[0]] = type[1]
                self.model_MAT_TYPE = list(parsed_MAT_TYPE.keys())
                self.blocks[b] = parsed_MAT_TYPE

                #Sanity check
                #Check that variable whose type is assigned in ENDO_TYPE is also declared in EXO_PARAM
                for var in self.model_MAT_TYPE:
                    if var not in self.model_ENDO_MAT:
                        raise ValueError('Syntax Error: Var \"'+var+'\" is assigned type in MAT_TYPE but is not declared in ENDO_MAT')

            elif b == 'ENDO_EQ':
                unparsed_ENDO_EQ = self.blocks[b]
                parsed_ENDO_EQ = {}
                # Sanity check
                flag = 0
                for eq in unparsed_ENDO_EQ:
                    if flag == 0:
                        if re.match('^\[.*\]$', eq):
                            pass
                        else:
                            raise ValueError('Syntax Error: '+eq)
                    else:
                        if re.match('^[^=]*=.*$', eq):
                            pass
                        else:
                            raise ValueError('Syntax Error: '+eq)
                    flag = (flag + 1) % 2
                ENDO_EQ_names = []
                flag = 0
                for eq in unparsed_ENDO_EQ:
                    if flag == 0:
                        equation = {}
                        for option in default_eq_options:
                            equation[option] = ''
                        option_split = re.sub('^\[|\]$| ', '', eq)
                        option_split_no_parenthesis = option_split
                        if option_split.count('(') != option_split.count(')'):
                            raise ValueError('Syntax Error: Invalid parenthesis in ' + eq)
                        if option_split.count('[') != option_split.count(']'):
                            raise ValueError('Syntax Error: Invalid brackets in ' + eq)
                        while '(' in option_split_no_parenthesis or ')' in option_split_no_parenthesis:
                            option_split_no_parenthesis = re.sub('\([^\(\)]*\)', repl, option_split_no_parenthesis)
                        option_split_idx = [c for c in range(len(option_split_no_parenthesis)) if option_split_no_parenthesis[c] == ',']
                        option_split = [opt for opt in split_by_idx(option_split, option_split_idx)]

                        for option in option_split:
                            opt = option.split('=', 1)
                            if opt[0] == 'condition':
                                if equation[opt[0]] != '':
                                    raise ValueError('Syntax Error: equation \"'+eq+'\" has more than one condition')
                                equation[opt[0]] = opt[1]
                            else:
                                if opt[0] == 'name':
                                    name = opt[1]
                                else:
                                    equation[opt[0]] = opt[1]
                    else:
                        equation['equation'] = eq
                        if name in ENDO_EQ_names:
                            raise ValueError('Another equation with name '+name+' was declared')
                        else:
                            parsed_ENDO_EQ[name] = equation
                        ENDO_EQ_names.append(name)

                    flag = (flag + 1) % 2

                # Sanity check
                # Check that all endogenous variables have an equation, otherwise throw a warning
                eq_vars = [parsed_ENDO_EQ[eq_var]['equation'].split('=')[0] for eq_var in list(parsed_ENDO_EQ.keys())]
                for var in self.model_ENDO_VARS:
                    if var not in eq_vars:
                        print('Warning: Endogenous variable '+var+' does not have an equation')

                # Check that the variable in the equation was declared
                for var in eq_vars:
                    if re.sub( '\[.*\]', '', var) not in self.model_ENDO_VARS+self.model_ENDO_MAT+self.model_EXO_PARAM+self.model_A_ITERATOR+self.model_A_GROUPS:
                        raise ValueError('Variable '+var+' is not declared in ENDO_VARS, ENDO_MAT, EXO_PARAM, as an Iterator, or as a Group')

                # Check that the number of parenthesis is consistent
                model_equations = [parsed_ENDO_EQ[eq_var]['equation'] for eq_var in list(parsed_ENDO_EQ.keys())]
                for eq in model_equations:
                    if eq.count('(') != eq.count(')'):
                        raise ValueError('Syntax Error: Invalid parenthesis in ' + eq)
                    if eq.count('[') != eq.count(']'):
                        raise ValueError('Syntax Error: Invalid brackets in ' + eq)

                # Check that variables and functions inside equation were respectively declared or are supported
                reg_expression = '\+|-|\*|\/|@|\(|\)|\]|\[|:|<|>|=|!|\.T| |\\\\'
                model_equations_raw = [re.sub(reg_expression, ',', eq) for eq in model_equations]
                model_equations_raw = [eq.split(',') for eq in model_equations_raw]
                model_equations_raw = [[term for term in eq if term != ''] for eq in model_equations_raw]
                for eq in model_equations_raw:
                    for term in eq:
                        if term in self.model_ENDO_VARS+self.model_ENDO_MAT+self.model_EXO_PARAM+self.model_A_ITERATOR+self.model_A_GROUPS+self.model_A_TAGS+self.hard_coded_names:
                            continue
                        elif term in self.supported_numpy_functions+self.supported_numpy_arguments:
                            continue
                        elif re.sub('[.]', '', term).isdigit():
                            continue
                        else:
                            raise ValueError('Variable or function \"'+term+'\" is not declared or is not supported')
                self.blocks[b] = parsed_ENDO_EQ

                # Make list of all equation names for following sanity checks
                self.model_EQ_NAMES = list(self.blocks[b].keys())

                # Check that none of the equation names are the same as the supported statements or numpy functions
                for eq_name in self.model_EQ_NAMES:
                    if eq_name in self.supported_numpy_functions + self.supported_numpy_arguments + self.supported_statements + self.hard_coded_names:
                        raise ValueError('Syntax Error: equation name \"'+eq_name+'\" is not allowed, please check supported functions, arguments, and hard coded names')

                # Check that none of the equation names contain [ or ]
                for eq_name in self.model_EQ_NAMES:
                    if '[' in eq_name:
                        raise ValueError('Syntax Error: Invalid character [ in name \"'+eq_name+'\"')
                    if ']' in eq_name:
                        raise ValueError('Syntax Error: Invalid character ] in name \"'+eq_name+'\"')

            elif b == 'STEPS':
                # Sanity checks
                # Check that all equations in STEPS are declared
                eq_in_STEPS = []
                for eq_name in self.blocks['STEPS']:
                    eq_name = re.sub('\[.*\]', '', eq_name)
                    if eq_name in self.model_EQ_NAMES:
                        eq_in_STEPS.append(eq_name)
                        continue
                    else:
                        check_statements = 0
                        for statement in self.supported_statements:
                            if re.match('^'+statement+'\(.*\){$', eq_name):
                                check_statements = 1
                        if check_statements == 0 and eq_name != '}' and eq_name not in self.supported_statements:
                            raise ValueError('Equation ' + eq_name + ' in STEPS is not declared in ENDO_EQ')
                        continue

                # Warn if a declared equation does not appear in STEPS
                for eq_name in self.model_EQ_NAMES:
                    if eq_name not in eq_in_STEPS:
                        print('Warning: Equation \"'+eq_name+'\" is not included in the STEPS')

                # Check that all conditions in while and for loops are correctly specified
                for eq_name in self.blocks['STEPS']:
                    for statement in self.supported_statements:
                        if statement in eq_name:
                            if re.match('^' + statement + '\(.+\){$', eq_name):
                                if statement == 'for':
                                    try:
                                        condition = re.search('\([^\(\),]*,[^\(\),]*\)', eq_name).group(0)
                                    except:
                                        raise ValueError('Syntax Error: \"' + eq_name + '\"')
                                    condition = re.sub('\(|\)', '', condition)
                                    condition = condition.split(',')
                                    for c in condition:
                                        if c in self.model_ENDO_VARS+self.model_ENDO_MAT+self.model_EXO_PARAM+self.model_A_ITERATOR+self.model_A_GROUPS+self.model_A_TAGS+self.hard_coded_names:
                                            continue
                                        else:
                                            raise ValueError('Variable \"'+c+'\" in statement \"'+eq_name+'\" is not declared')
                            elif statement == 'break':
                                continue
                            elif statement == 'quit':
                                continue
                            else:
                                raise ValueError('Syntax Error: Malformed statement \"'+eq_name+'\"')
            else:
                raise ValueError('Block class \"'+b+'\" not recognized')
        # Finished parsing mod file

        compiled_file = []

        # Import libraries
        compiled_file.append('import numpy as np\n')
        compiled_file.append('from numba import njit\n')

        compiled_file.append('\n')

        # Start writing model function
        if self.blocks['SETTINGS']['numba'] == 'True':
            compiled_file.append('@njit(cache = True)\n')
        compiled_file.append('def run(iterations, params, verbose):\n')

        # Sort agents to make output consistent and more organized
        sorted_agents = list(self.blocks['AGENTS'].keys())
        sorted_agents.sort()

        #Set settings
        compiled_file.append('    # SET SETTINGS\n')
        if self.blocks['SETTINGS']['isclose'] != '0':
            compiled_file.append('    numpy_prec = '+self.blocks['SETTINGS']['isclose']+'\n')

        # Initialize database
        compiled_file.append('    # BUILDING DATABASES\n')
        for a in sorted_agents:
            num = self.blocks['AGENTS'][a]['num']
            num_var = len(self.blocks['ENDO_VAR'][a])
            compiled_file.append('    ' + a + ' = np.zeros((iterations, '+str(num)+', '+str(num_var)+'))\n')
        compiled_file.append('\n')

        #Build reverse dictionary of endo_var, mapping the var to the agent
        self.endo_var_agent_map = {}
        for a in sorted_agents:
            for var_i, var in enumerate(self.blocks['ENDO_VAR'][a]):
                var_map = {}
                var_map['AGENT'] = a
                var_map['INDEX'] = var_i
                self.endo_var_agent_map[var] = var_map

        # Set parameters
        compiled_file.append('    #SETTING PARAMETERS\n')
        # 1. Initialize tags
        for a in sorted_agents:
            tag = self.blocks['AGENTS'][a]['tag']
            num = self.blocks['AGENTS'][a]['num']
            compiled_file.append('    ' + tag + ' = ' + num + '\n')
        # 2. Initialize iterators
        for a in sorted_agents:
            iterator = self.blocks['AGENTS'][a]['iterator']
            val = 0
            for i in iterator:
                compiled_file.append('    ' + i + ' = ' + str(val) + '\n')

        # Sort exo_param to make output consistent and more organized
        self.sorted_exo_param = list(self.blocks['EXO_PARAM'].keys())
        self.sorted_exo_param.sort()

        # 3. Initialize exogenous parameters
        self.params = []
        for p_i, p in enumerate(self.sorted_exo_param):
            val = self._get_formula(self.blocks['EXO_PARAM'][p], True, init=True)
            if '.' in val:
                self.params.append(float(val))
            else:
                self.params.append(int(val))
            compiled_file.append('    #' + p + ' = '+val+'\n')
            compiled_file.append('    ' + p + ' = params['+str(p_i)+']\n')
        self.params = np.array(self.params)
        
        # Sort endo_mat to make output consistent and more organized
        sorted_endo_mat = list(self.blocks['ENDO_MAT'].keys())
        sorted_endo_mat.sort()
        compiled_file.append('    \n')

        # 3. Initialize matrix
        compiled_file.append('    #SETTING UP MATRIX\n')
        for m in sorted_endo_mat:
            val = self._get_formula(self.blocks['ENDO_MAT'][m]['eq'], True, init=True)
            compiled_file.append('    ' + m + ' = ' + str(val) + '\n')
        compiled_file.append('    \n')

        # 5. Set matrix types
        compiled_file.append('    #SETTING UP MATRIX TYPE\n')
        sorted_endo_type = list(self.blocks['MAT_TYPE'].keys())
        sorted_endo_type.sort()
        for endo_type in sorted_endo_type:
            var = endo_type
            val = self.blocks['MAT_TYPE'][endo_type]
            compiled_file.append('    ' + var + ' = ' + var + '.astype(np.' + str(val) + ')\n')
        compiled_file.append('\n')

        # Initialize values of endogenous variables
        compiled_file.append('    #SETTING INITIAL VALUES\n')
        sorted_endo_init = list(self.blocks['ENDO_INIT'].keys())
        sorted_endo_init.sort()
        for var in sorted_endo_init:
            lhs = self._get_formula(var, False, init=True)
            rhs = self._get_formula(self.blocks['ENDO_INIT'][var], True, init=True)
            compiled_file.append('    #'  + var + '\n')
            compiled_file.append('    ' + lhs + ' = ' + rhs + '\n')
        compiled_file.append('\n')

        # Extend database for the entire period
        compiled_file.append('    #Extend database for entire period\n')
        compiled_file.append('    for t in range(iterations):\n')
        for agent in sorted_agents:
            compiled_file.append('        ' + agent + '[t, :, :] = ' + agent + '[0, :, :]\n')
        compiled_file.append('\n')

        #Start model loop
        compiled_file.append('    #STEPS OF THE MODEL\n')
        compiled_file.append('    for t in range(iterations):\n')
        compiled_file.append('        if t == 0:\n')
        compiled_file.append('            continue\n')
        compiled_file.append('        if verbose == True:\n')
        compiled_file.append('            print(t)\n')

        whiteSpaceCounter = 2
        whiteSpace = '    '

        #DEBUG TRANSFER ALL VARIABLE VALUES TO NEXT ITERATION
        for var in self.model_ENDO_VARS:
            lhs = self._get_formula(var, False, lag=0)
            rhs = self._get_formula(var, True, lag=1)
            compiled_file.append(whiteSpaceCounter * whiteSpace + lhs + ' = ' + rhs + '\n')


        for step in self.blocks['STEPS']:
            if re.sub('\[.*\]', '' , step) in self.model_EQ_NAMES:
                iterators = re.split('\[|\]', step)
                iterators = [p for p in iterators[1:] if p!='']
                step = re.sub('\[.*\]', '' , step)
                eq = self.blocks['ENDO_EQ'][step]['equation']
                eq_split = eq.split('=', 1)
                lhs = eq_split[0]
                rhs = eq_split[1]
                # There are several things to parse and to keep in mind in the following section.
                # The goal is to have a rhs string which contains the right hand side part of the equation with conditions included
                # This string can then easily be changed into its numpy form and can also be commented above the equation for debugging purposes

                #Regarding conditions, we need to parse conditional statements which may contain conditions within conditions
                #The above is solved by iterating through thestring and pushing subgroups within groups whenever we encounter a ( or )
                #The subgroups in the output can then be split along the && and || operators, which will give the computable statement
                condition = self.blocks['ENDO_EQ'][step]['condition']

                #Iterating through the condition, and creating subgroups out of string in parenthesis
                groups = []
                depth = 0
                try:
                    for char in condition:
                        if char == '(' or char == '[':
                            push(char, groups, depth)
                            push([], groups, depth)
                            depth += 1
                        elif char == ')' or char == ']':
                            depth -= 1
                            push(char, groups, depth)
                        else:
                            push(char, groups, depth)
                except IndexError:
                    raise ValueError('Syntax Error: Parentheses mismatch in condition \"'+condition+'\"')
                if depth > 0:
                    raise ValueError('Syntax Error: Parentheses mismatch in condition \"'+condition+'\"')
                #Recursively iterating out of the subgroups, making the necessary && || changes at the same time
                condition = rec_iter_groups(groups)
                if len(condition)>0:
                    not_condition = 'abs(('+condition+')-1)'
                    rhs = '(' + condition + ')' +' * ' + '(' + rhs + ')' + ' + ' + '(' + not_condition + ' * ' + '(' + lhs + ')' + ')'

                compiled_file.append(whiteSpaceCounter*whiteSpace+'# '+lhs+' = '+rhs+'\n')
                lhs = self._get_formula(lhs, False, lag=0, iterators=iterators)
                rhs = self._get_formula(rhs, True, lag=0, iterators=iterators)
                compiled_file.append(whiteSpaceCounter*whiteSpace+lhs+' = '+rhs+'\n')
                if self.blocks['SETTINGS']['show_values'] == 'True':
                    compiled_file.append(whiteSpaceCounter*whiteSpace+'if t == t:\n')
                    compiled_file.append((whiteSpaceCounter+1)*whiteSpace+'print(\''+str(step)+'\')\n')
                    compiled_file.append((whiteSpaceCounter+1)*whiteSpace+'print('+lhs+')\n')

            elif step == '}':
                whiteSpaceCounter = whiteSpaceCounter-1
            elif any([re.match('^' + statement + '\(.+\){$', step) for statement in self.supported_statements]) or 'break' or 'quit' in step:
                for statement in self.supported_statements:
                    if re.match('^' + statement + '\(.+\){$', step):
                        break
                if 'break' in step:
                    statement = 'break'
                if 'quit' in step:
                    statement = 'quit'
                if statement == 'if' or statement == 'while':
                    condition = re.search('\(.*\)', step).group(0)[1:-1]
                    groups = []
                    depth = 0
                    try:
                        for char in condition:
                            if char == '(' or char == '[':
                                push(char, groups, depth)
                                push([], groups, depth)
                                depth += 1
                            elif char == ')' or char == ']':
                                depth -= 1
                                push(char, groups, depth)
                            else:
                                push(char, groups, depth)
                    except IndexError:
                        raise ValueError('Syntax Error: Parentheses mismatch in condition \"' + condition + '\"')
                    if depth > 0:
                        raise ValueError('Syntax Error: Parentheses mismatch in condition \"' + condition + '\"')
                    condition = rec_iter_groups(groups)
                    condition = self._get_formula(condition, True, lag=0)
                    compiled_file.append(whiteSpaceCounter * whiteSpace + statement + ' ' + condition + ':\n')
                    whiteSpaceCounter = whiteSpaceCounter+1
                elif statement == 'for':
                    condition = re.search('\([^\(\),]*,[^\(\),]*\)', step).group(0)[1:-1]
                    condition = condition.split(',')
                    compiled_file.append(whiteSpaceCounter * whiteSpace + statement + ' ' + condition[0] + ' in ' + condition[1] + ':\n')
                    whiteSpaceCounter = whiteSpaceCounter+1

                elif statement == 'break':
                    compiled_file.append(whiteSpaceCounter * whiteSpace + statement + '\n')
                elif statement == 'quit':
                    compiled_file.append(whiteSpaceCounter * whiteSpace + statement + '()\n')
                else:
                    raise ValueError('Syntax Error: Statement is not implemented \"' + step + '\"')
            else:
                raise ValueError('Syntax Error: Cannot understand step \"' + step + '\"')

        # TEMPORARY UNTIL NICER SOLUTION
        agents = list(self.blocks['ENDO_VAR'].keys())
        for agent in agents:
            db_names = self.blocks['ENDO_VAR'][agent]
            if len(db_names) == 0:
                db_names = ['']
            compiled_file.append('    ' + agent + '_db = '+str(db_names) + '\n')
        all_db = '[' + agents[0] + "".join([', ' + a for a in agents[1:]]) + ']'
        all_db_names = '['+ agents[0] +'_db' + "".join([', '+a+'_db' for a in agents[1:]]) + ']'
        compiled_file.append('    ' + 'db = ' + str(all_db) + '\n')
        compiled_file.append('    ' + 'db_name = ' + str(all_db_names) + '\n')
        compiled_file.append('    ' + 'return db, db_name\n')



        for func in self.numba_func.include:
            function = self.numba_func[func].split('\n')
            for l in function:
                compiled_file.append(l+'\n')

        if writeModel == True:
            with open('compiledModel.py', 'w') as f:
                for l in compiled_file:
                    f.write(l)
        else:
            print('Warning: Model loaded from cache')
            
        #Test model simulation and compile with numba if model changed.
        self.run(1, plot=False, verbose = False)

    def _get_formula(self, eq, rhs, init=False, lag=0, iterators=[]):
        """
        Returns parsed equation.
        
        """
        
        # Store equation for parsing purposes
        eq_parse = eq
        # Parse elements of the equation
        reg_expression = '\+|-|\*|\/|@|\(|\)|\]|\[|:|<|>|=|!|\.T| |\\\\'
        eq_parse = re.sub(reg_expression, ',', eq_parse)
        eq_parse = eq_parse.split(',')
        eq_parse = [term for term in eq_parse if term != '']
        eq_parse = list(set(eq_parse))
        for term in eq_parse:
            if term in self.model_ENDO_VARS:
                eq = replace_elem(eq, term, self._get_variable(term, rhs, init=init, lag=lag, iterators=iterators))
            elif term in self.numba_func._all_functions():
                self.numba_func._add_to_include(term)
                continue
            elif term in self.supported_numpy_functions:
                eq = replace_elem(eq, term, 'np.'+term)
            elif term in self.supported_numpy_arguments:
                continue
            elif re.sub('[.]', '', term).isdigit():
                continue
            elif term in self.model_ENDO_MAT:
                eq = replace_elem(eq, term, self._get_matrix(term, rhs, lag=lag, iterators=iterators))
            elif term in self.model_EXO_PARAM + self.model_A_ITERATOR + self.model_A_GROUPS + self.model_A_TAGS + self.hard_coded_names:
                continue
            else:
                raise ValueError('Syntax Error: term \"'+term+'\" is not valid')
        return eq

    def _get_variable(self, var, rhs, init=False, lag=0, iterators=[]):
        """
        Returns endo_var used in the equation with the correct indexing and lag.
        
        """
    
        #We use the reverse mapping to directly retrieve the agent of a variable, and consequently to which database it refers to.
        agent = self.endo_var_agent_map[var]['AGENT']
        index = self.endo_var_agent_map[var]['INDEX']
        if int(self.blocks['AGENTS'][agent]['num']) > 1:
            iterator = ':'
        else:
            iterator = self.blocks['AGENTS'][agent]['iterator'][0]
        append_group = ''
        for i in iterators:
            if i in self.blocks['AGENTS'][agent]['iterator']:
                iterator = i
            if i in self.blocks['AGENTS'][agent]['group']:
                iterator = ':'
                append_group = '.take('+i+')'
        if rhs == True:
            #prec_arg_l = '(np.floor('
            #prec_arg_r = '*numpy_prec)/numpy_prec)'
            prec_arg_l = ''
            prec_arg_r = ''
        else:
            prec_arg_l = ''
            prec_arg_r = ''
        if init == True:
            return prec_arg_l+agent+'[0, '+iterator+', '+str(index)+']'+append_group+prec_arg_r
        else:
            if lag == 0:
                return prec_arg_l+agent+'[t, '+iterator+', '+str(index)+']'+append_group+prec_arg_r
            else:
                return prec_arg_l+agent + '[t-'+str(lag)+', '+iterator+', ' + str(index) + ']'+append_group+prec_arg_r

    def _get_matrix(self, var, rhs, lag=0, iterators=[]):
        """
        Returns endo_mat used in the equation with the correct indexing and lag.
        
        """
        #We use the reverse mapping to directly retrieve the agent of a variable, and consequently to which database it refers to.
        num_iterators = len(self.blocks['ENDO_MAT'][var]['tags'])
        if lag == 0:
            t = 't'
        else:
            t = 't-'+str(lag)

        if rhs == True:
            #prec_arg_l = '(np.floor('
            #prec_arg_r = '*numpy_prec)/numpy_prec)'
            prec_arg_l = ''
            prec_arg_r = ''
        else:
            prec_arg_l = ''
            prec_arg_r = ''

        if len(iterators) == 0:
            return prec_arg_l+var+'[:'+' ,:'*(num_iterators-1)+']'+prec_arg_r
        else:
            #Check that iterators are either declared as agent iterators or are endo_mat
            for i in iterators:
                if i in self.model_A_ITERATOR:
                    continue
                elif i in self.model_ENDO_MAT:
                    continue
                elif i in self.model_EXO_PARAM:
                    continue
                else:
                    raise ValueError('Syntax Error: Iterator \"'+i+'\" is not declared')
            #Check that if iterator is an agent iterator, that they conform with the iterators provided when matrix was declared
            for i in iterators:
                if i in self.model_A_ITERATOR and i not in self.blocks['ENDO_MAT'][var]['tags']:
                    raise ValueError('Syntax Error: Agent iterator \"'+i+'\" was not provided for this matrix when it was declared')
                else:
                    continue

            #Generate matrix indexing
            mat_index = self.blocks['ENDO_MAT'][var]['tags']
            mat_index = [i if i in iterators else ':' for i in mat_index]

            #Check that order of iterators follows dimensions of matrix
            for ii, i in enumerate(mat_index):
                if i in self.model_A_ITERATOR and not(i == self.blocks['ENDO_MAT'][var]['tags'][ii] or i == ':'):
                    raise ValueError('Syntax Error: Agent iterator \"'+i+'\" in STEPS was provided in a different dimension from the one declared in ENDO_MAT')
                else:
                    continue
            return prec_arg_l+var+'['+mat_index[0]+"".join([', '+i for i in mat_index[1:]])+']'+prec_arg_r

    def plot_normalize(self, normalization_input):
        """
        Modifies 'private' variable which specifies variables to normalize during plotting.
        
        Parameters
        ----------
        normalization_input : dict
            Dictionary with normalization input.
        """
        
        self._plot_normalization = normalization_input

    def run(self, iterations, plot=False, params = np.array([]), verbose = False):
        """
        Runs simulation for selected amount of iterations.
        
        Parameters
        ----------
        iterations : int
            Number of times to iterate the model.
            
        plot : bool
            True will plot all the variables in the model in a folder named after the model file and the selected number of iterations.
            
        params : ndarray
            Parametrization to use in the model. Default will use parametrization currently saved in the preprocessor object.
            
        verbose : int
            1 will print the current iteration during simulation.
            
        Returns
        ----------
        
        dbase : dseries
            dseries object containing the simulated data.
        """
    
    
        from compiledModel import run
        if len(params) == 0:
            params = self.params
        dbase = run(iterations, params, verbose)
        dbase = dseries(dbase[0], dbase[1])
        if plot == True:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(10, 10))
            folder_to_save_plots = self.modfile+str(iterations)+'/'
            if os.path.exists(folder_to_save_plots):
                pass
            else:
                os.mkdir(folder_to_save_plots)
            for n in dbase.names:
                if n in list(self._plot_normalization.keys()):
                    plt.plot(dbase[n]/dbase[self._plot_normalization[n]])
                    plt.suptitle(n)
                    plt.savefig(folder_to_save_plots + n + '.png')
                    plt.clf()
                else:
                    plt.plot(dbase[n])
                    plt.suptitle(n)
                    plt.savefig(folder_to_save_plots + n + '.png')
                    plt.clf()
        return dbase

    def estimate(self, dbase, initial_params, estimated_vars, iterations = 1000, batch = 10, start = 0, step=0.0001, parallel = False):
        """
        Estimates selected parameters on selected variables using a Metropolis-Hastings algorithm with simulated maximum likelihood.
    
        Parameters
        ----------
        dbase : dseries
            Data on which the model is to be estimated.
            
        initial_params : dict
            Selected parameters to estimate and their initial guesses.
            
        estimated_vars : list of str
            Selected variables used to compute simulated maximum likelihood.
            
        iterations : int
            Number of iterations of the metropolis algorithm.
            
        batch : int
            Number of times the model is to be ran on a single parametrization.
            
        step : float
            Metropolis-Hastings step size.
            
        parallel : bool
            True will run the batch of models using multiple processors for the current parametrization.
            False will run the model one by one.
            
        Returns
        -------
        params : ndarray
            The estimated parametrization at the end of the algorithms' run.
                
        """
        def SMLE(y, yhat, batch):
            """ 
            Computes simulated maximum likelihood for a single variable.
            
            """
            
            def Kn(ye, eta, N):
                return K(ye/np.sqrt(eta), eta, N)/np.sqrt(eta)

            def K(psi, eta, N):
                return (1/(np.sqrt(2*np.pi)*np.std(psi))*N)*np.exp(-(abs(psi)**2)/(2*(np.std(psi)**2)))
            def likelihood(pt):
                return np.sum(np.log(pt))
            N = batch
            sigmahat = np.std(yhat, axis=0)
            eta = ((4/(3*N))**(1/5))*sigmahat
            ye = y - yhat
            pt = 1/N * np.sum(Kn(ye, eta, N), axis=1)
            return likelihood(pt)
            
        import matplotlib.pyplot as plt
        if parallel == True:
            import multiprocessing as mp
            pool = mp.Pool(mp.cpu_count())
        run = 100
        for n in dbase.names:
            dbase[n] = np.mean(dbase[n], axis=1).reshape((run, 1))
        for n in dbase.names:
            if n not in estimated_vars:
                dbase.pop(n)
               
        param_names = list(initial_params.keys())
        old_loglik = -np.inf
        params_prev = initial_params

        all_param = []
        all_SMLE = []
        for t in range(iterations):
            print(t)
            params_eval = params_prev
            for p in (param_names):
                params_eval[p] = np.random.normal(params_eval[p], step)
            for p in (param_names):
                self.params[self.sorted_exo_param.index(p)] = params_eval[p]
            if parallel == True:
                args = [[run, self.params, False] for b in range(batch)]
                with mp.Pool() as pool:
                    output = pool.map(run_parallel, args)
            for b in range(batch):
                if b == 0:
                    if parallel == True:
                        dbase_new = output[0]
                    else:
	                    dbase_new = self.run(run, params = self.params)
                    
                    for n in dbase_new.names:
	                    if n not in estimated_vars:
	                        dbase_new.pop(n)
                    for n in estimated_vars:
                        dbase_new[n] = np.mean(dbase_new[n], axis=1).reshape((run, 1))
                else:
                    if parallel == True:
                        dbase_tmp = output[b]
                    else:
                        dbase_tmp = self.run(run, params = self.params)
                    for n in estimated_vars:
                        dbase_new[n] = np.concatenate((dbase_new[n], np.mean(dbase_tmp[n], axis=1).reshape((run, 1))), axis=1)
            self.params_prev = self.params
            new_loglik = 0
            for n in estimated_vars:
                new_loglik = new_loglik + SMLE(dbase[n], dbase_new[n], batch)
            print(new_loglik)
            print(params_eval)
            if new_loglik>old_loglik:
                params_prev = params_eval
            else:
                u = np.random.uniform(0,1)
                if u < np.exp(new_loglik-old_loglik):
                    params_prev = params_eval
                else:             
                    params_prev = params_prev
            

            all_param.append(params_eval.copy())
            all_SMLE.append(new_loglik)
            if t % 100 == 0 and t != 0:
                parsed_all_param = {}
                for p in param_names:
                    parsed_param = []
                    for param in all_param:
                        parsed_param.append(param[p])
                    parsed_all_param[p] = parsed_param
                fig, axs = plt.subplots(len(param_names), len(param_names), figsize=(15,15))
                cmap = plt.get_cmap('RdYlGn')
                colors = cmap((np.array(all_SMLE) - min(all_SMLE))/ np.array(all_SMLE).ptp())
                for y, p_y in enumerate(param_names):
                    for x, p_x in enumerate(param_names):
                        axs[x,y].scatter(parsed_all_param[p_y], parsed_all_param[p_x], c = colors, cmap = cmap)
                for y_i, y in enumerate(axs):
                    for x_i, x in enumerate(y):
                        x.set(xlabel=param_names[x_i], ylabel=param_names[y_i])
                for ax in axs.flat:
                    ax.label_outer()
                plt.savefig('metropolis_distris.png')
                plt.clf()

                fig, axs = plt.subplots(len(param_names), 1, figsize=(15,15))
                for y, p_y in enumerate(param_names):
                    axs[y].plot(parsed_all_param[p_y])
                for y_i, y in enumerate(axs):
                    y.set(title=param_names[y_i])
                plt.savefig('metropolis_res.png')
                plt.clf()
            
        return params_prev

    class numba_functions:
        """
        Object containing user built functions.
        !!!Temporary solution until something better!!!

        Attributes:
        ----------
        func : dict
        Dictionary containing user made functions.
        Example:
        
        >>> self.func['isclose'] = \"\"\"@njit(cache = True)
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    diff = np.abs(b - a)
    return np.fmin(np.fmin((diff <= np.abs(rel_tol * b)) + (diff <= np.abs(rel_tol * a)), 1) + (diff <= abs_tol), 1)\"\"\"
        

        """
        def __init__(self):
            self.func = dict()
            self.include = []
            self.func['isclose'] ="""@njit(cache = True)
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    diff = np.abs(b - a)
    return np.fmin(np.fmin((diff <= np.abs(rel_tol * b)) + (diff <= np.abs(rel_tol * a)), 1) + (diff <= abs_tol), 1)"""


            self.func['size'] = """@njit(cache = True)
def size(x):
    return np.sum(x**0)
        """

            self.func['min'] = """@njit(cache = True)
def min(x, initial=np.inf):
    for i in x:
        if i<initial:
            initial = i
    return initial
        """

            self.func['nansum'] = """@njit(cache = True)
def nansum(x, axis = 0):
    if axis == 1:
        x = x
    elif axis == 0:
        x = x.T
    else:
        raise ValueError('Axis value above 1 are currently unsupported')
    temp_nan_sum = np.zeros((x.shape[0]))
    for nan_index_i in range(x.shape[0]):
        temp_nan_sum[nan_index_i] = np.nansum(x[nan_index_i])
    return temp_nan_sum"""

        def __getitem__(self, func_name):
            """
            Returns function 'func_name' from dictionary.
            """
            
            return self.func[func_name]

        def _all_functions(self):
            """
            Returns name of all functions in dictionary.
            """
            
            return list(self.func.keys())

        def _add_to_include(self, func_name):
            """
            Adds name of a function in the dictionary if found in the model file.
            Used to only include custom functions if called in the model file.
            """
            
            self.include.append(func_name)
            self.include = list(set(self.include))


class dseries:
    """
    Database object. Used primarily to contain model simulation output and to compute simulated maximum likelihood during estimation.
    ...
    
    Parameters
    ----------
    db : list of ndarray
        List containing the matrix for each agent in the model
        
    db_name : list of list of str
        List containing a list for each agent of the variable names.
        
    Example: 
    
    >>> dbase = dseries([BANK, FIRMS],[['OMEGAB', 'LAMBDAB'], ['OMEGAF', 'OMEGAB'])
    
    >>> dbase = dseries([],[])
    >>> dbase.load('dbase.dat')
    
    
    Attributes
    ----------
    names : list of str
        Contains the name of all the variables in the database
        
    Example:
    
    >>> for name in dbase.names:
            print(name)
            print(dbase[name])
        
    Methods
    ----------
    save(filename = 'dbase')
        Saves the database in a pickled file named after 'filename'.
        Note that .dat is added automatically after 'filename'.
        
    load(filename = 'dbase.dat')
        Loads the database saved in 'filename'.
        The previous database is overwritten with the new data.
        Note that .dat is required in the 'filename' input.
        
    pop(var_name = 'OMEGAB')
        Deletes variable 'var_name' from the database.
    
    
    Returns
    -------
    out : dseries
        A dseries object containing the specified data.
    """
    
    def __init__(self, db, db_name):
        self.data = []
        self.names = []
        for i in range(len(db)):
            if db[i].size == 0:
                continue
            for n in range(db[i].shape[2]):
                self.data.append(db[i][:,:,n])
                self.names.append(db_name[i][n])

    def __getitem__(self, var_name, lag = 0):
        """
        Returns ndarray containing selected data.
        
        Example:
        
        >>> dbase['OMEGAB']
        
        array([[2, 3]])

        """
        
        if isinstance(var_name, tuple):
            lag = var_name[1]
            var_name = var_name[0]
        if var_name in self.names:
            var_index = self.names.index(var_name)
            var_data = self.data[var_index][:self.data[var_index].shape[0]-lag,:]
            lag_nans = np.empty((lag,self.data[var_index].shape[1]))
            lag_nans[:,:] = np.nan
            var_output = np.concatenate((lag_nans, var_data))
            return var_output
        raise ValueError('Variable "'+var_name+'" does not exist in database')

    def  __setitem__(self, var_name, value):
        """
        Adds or replaces variable 'var_name' with 'value'.
        Value must be ndarray.
        
        """
    
    
        if var_name in self.names:
            self.data[self.names.index(var_name)] = value
        else:
            if type(value) != np.ndarray:
                raise ValueError('Values added to database must be np.array')
            self.names.append(var_name)
            self.data.append(value)

    def save(self, filename):
        """
        Saves dseries data to 'filename'.
        
        """
        dat = [self.data, self.names]
        with open(filename+'.dat', 'wb') as f:
            pickle.dump(dat, f)
        print("Database saved to: "+filename+'.dat')

    def load(self, filename):
        """
        Loads dseries data stored in 'filename'.
        
        """
        with open(filename, 'rb') as f:
            dat = pickle.load(f)
        self.data = dat[0]
        self.names = dat[1]
        return self

    def pop(self, var_name):
        """
        Deletes variable in 'var_name' from dseries object.
        
        """
        if var_name in self.names:
            i = self.names.index(var_name)
            self.names.pop(i)
            self.data.pop(i)
        else:
            raise ValueError('Variable '+var_name+' does not exist in dataset')
        return self

    def __add__(self, other):
        """
        EXPERIMENTAL
        
        Concatenates data from one dseries with that of another dseries for each variable they have in common.
        
        """
        for n in self.names:
            self[n] = np.concatenate((self[n], other[n]), axis=0)
        return self


def repl(m):
    """
    Generates string of # of length m.
    Function used to replace text inside parenthesis during equation parsing so
    that regex does not split content between parenthesis.
    
    """
    return '#' * len(m.group())

def split_by_idx(s, list_of_indices):
    """
    Splits list relative to a list of indices.
    
    """
    if len(list_of_indices) > 0:
        left, right = 0, list_of_indices[0]
        yield s[left:right]
        left = right+1
        for right in list_of_indices[1:]:
            yield s[left:right]
            left = right
    else:
        left = 0
    yield s[left:]

def push(obj, l, depth):
    """
    Helper function used in rec_iter_groups.
    
    """
    while depth:
        l = l[-1]
        depth -= 1
    l.append(obj)

def rec_iter_groups(groups):
    """
    Function used to safely parse conditional statements so that they respect parenthesis order.
    
    """

    while ~all(isinstance(group, (str)) for group in groups):
        if all(isinstance(group, (str)) for group in groups):
            groups = ''.join(groups)
            if '||' in groups or '&&' in groups:
                groups = groups.split('&&')
                new_groups = groups[0]
                for group in groups[1:]:
                    new_groups = '('+new_groups+') * (' +group+')'
                groups = ''.join(new_groups)
                groups = groups.split('||')
                new_groups = groups[0]
                for group in groups[1:]:
                    new_groups = 'fmin(('+new_groups+') + (' +group+'), 1)'
                groups = ''.join(new_groups)
            return groups
        else:
            for i in range(len(groups)):
                if type(groups[i]) == str:
                    continue
                else:
                    groups[i] = rec_iter_groups(groups[i])

def replace_elem(s, elem, elem_):
    """
    Function used to replace a variable with its ndarray form.
    Convoluted but necessary due to the fact that naively replacing strings
    could mistake a variable with another with a similar name (OMEGAB becoming BANK[t,1,:],
    mistaking OMEGAB_Pm for OMEGAB, and thus returning BANK[t,1,:]_Pm)
    
    """
    validStoppers = [' ', '+', '-', '*', '@', '/', '(', ')',',', '=', '!', '>', '<', '.T', '[', ']', ':']
    maxStoppersLen = max(len(x) for x in validStoppers)
    i = 0
    length_S = len(s)
    while i < (length_S):
        cur_s = s[i:i+len(elem)]
        if cur_s == elem:
            lhs = [s[max(i-1-Si, 0):max(0,i)] for Si in range(maxStoppersLen)]
            rhs = [s[min(i+len(elem), len(s)-1):min(i+len(elem)+Si+1, len(s))] for Si in range(maxStoppersLen)]
            if (i-1<0 and (len(set(rhs) & set(validStoppers)) > 0))\
                    or (i+len(elem)==len(s) and (len(set(lhs) & set(validStoppers)) > 0))\
                    or ((len(set(lhs) & set(validStoppers)) > 0) and (len(set(rhs) & set(validStoppers)) > 0))\
                    or (i+len(elem)==len(s) and i-1<0):

                s_lhs = s[:i]
                s_rhs = s[i+len(elem):]
                s = s_lhs+elem_+s_rhs
                length_S = len(s)-len(elem)+1
                i = i +len(elem_)-1
        i = i + 1
    return s

def run_parallel(arg):
    """
    Function used to run parallel model. Must be outside object due
    to restrictions with multiprocessing regarding object pickling.
    
    """

    from compiledModel import run
    iterations = arg[0]
    params = arg[1]
    verbose = arg[2]
    dbase = run(iterations, params, verbose)
    dbase = dseries(dbase[0], dbase[1])
    return dbase

if __name__ == "__main__":

    #Preprocess model.txt and construct ABM object for simulation and estimation
    model = ABM('model.txt', cache=True)
    
    #Set variables to be normalized during plotting after model simulation
    model.plot_normalize({'BADB':'Pm','BETAB':'Pm','BETABREAL':'Pm',
                          'BETABRES':'Pm','BETAF':'Pm','BETAFREAL':'Pm',
                          'BETAFRES':'Pm','C':'Pm','CRES':'Pm',
                          'DELTAH':'Pm','DIV':'Pm','DIV1':'Pm',
                          'GAMMAB':'Pm','GAMMAF':'Pm','GAMMAFres':'Pm',
                          'Gw':'Pm','INTB':'Pm','INTCB':'Pm',
                          'INTCBb':'Pm','INTF':'Pm','INTPDEBTB':'Pm',
                          'LAMBDAB':'Pm','LAMBDAF':'Pm','M':'Pm',
                          'OMEGAB':'Pm','OMEGAF':'Pm','OMEGAF0':'Pm',
                          'OMEGAF':'Pm','OMEGAH':'Pm','PIGRB':'Pm',
                          'PIGRF':'Pm','PSALVA':'Pm','Pdebt':'Pm',
                          'PdebtCB':'Pm','Pdef':'Pm','Ptot':'Pm',
                          'Ptot2':'Pm','Ptot3':'Pm','SPESA':'Pm',
                          'TaxB':'Pm','TaxF':'Pm','TaxH':'Pm',
                          'TaxPatrF':'Pm','TaxPatrH':'Pm',
                          'Wm':'Pm','Wm0':'Pm','Wtot':'Pm'})
                      
    #Run model
    dbase = model.run(iterations = 1000, plot=True, verbose = 1)
    
    #Save simulation results
    dbase.save('dbase')
    
    #Set parameters to be estimated and initial guesses
    params = {'adjF': 0.1,
              'adjFprice': 0.1,
              'adjFleva': 0.1,
              'adjB': 0.1,
              'adjH': 0.1,
              }
              
    #Set variables used to estimate parameters
    estimated_vars = ['w', 'rD1', 'levat', 'price', 'd', 'rB1', 'db', 'rD']
    
    #Run estimation
    #Only run the following with a substantial computer
    #Need more testing but this routine reuqires at least 12gb of ram for this particular model.
    from psutil import virtual_memory
    mem = virtual_memory()
    if mem.total/1000000000<15:
        raise ValueError('Minimal Ram requirement check failed, this routine requires more memory to safely run.')
    else:
        params = model.estimate(dbase, initial_params=params, estimated_vars=estimated_vars, 
                            iterations = 100000, start = 0, parallel = True)

