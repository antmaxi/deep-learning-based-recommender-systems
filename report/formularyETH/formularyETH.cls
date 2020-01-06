%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Authors:
% Georg R. Pollak 2016
%
% License:
% The MIT License (see included LICENSE file)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{formularyETH/formularyETH}[2016/10/11 cheat sheet/formulary class v2.0]
%========================================================================================
%	CLASS CONFIGURATION
%========================================================================================

\RequirePackage{xstring}
% More generic control statements for class authors
\RequirePackage{etoolbox}
% ifthen conditionals TODO maybe we do not need this because of etoolbox
\RequirePackage{xifthen} 
% Key value options for class options
\RequirePackage{kvoptions}
\SetupKeyvalOptions {
  family = report,
  prefix = report@,
  setkeys = \setkeys
}
% More Controll/Macro statements, optional arguments
%----------------------------------------------------------------------------------------
% Defining more generic commands with xpars	
%----------------------------------------------------------------------------------------
\RequirePackage{xparse} 
% \DeclareDocumentCommand\comand{ arg1 arg1 ... } { Command definition }
% Mandatory arguments
    % m: mandatory {} (can be multiple {}{})
% Optional arguments
    % s: star 
    % o: optional [] (-NoValue- marker if none is given)
    % O: As for o , but returns <default> if no value is given. Should be given as O{<default>}
    % d: optional del1 del2: An optional argument which is delimited by <char1> and <char2>, given as follows: d<char1><char2>.
    % As with o , if no value is given the special marker -NoValue- is returned
    % D: As for d , but returns <default> if no value is given. Should be given as G<char1><char2>{<default>}
    % g: optional {} an optional argument which is delimited by {}
    %    As with o , if no value is given the special marker -NoValue- is returned
    % G: As for g , but returns <default> if no value is given. Should be given as G{<default>}

%   \IfBooleanTF{#1}{if True}{if False}
%   \IfNoValueTF {<argument>} {<true code>} {<false code>}
%   \IfNoValueT {<argument>} {<true code>}
%   \IfNoValueF {<argument>} {<false code>}
% ==============================================================================
% Similar to xparse but not as flexible (in order to be compatible with commands that still rely on it) 
\RequirePackage{xargs} 

%----------------------------------------------------------------------------------------
%	DOCUMENT OPTIONS
%----------------------------------------------------------------------------------------
\DeclareStringOption[scrartcl]{baseClass}[scrartcl]
% Paper Orientation - DEFAULT: portrait 
\DeclareBoolOption{landscape}
\DeclareBoolOption{debug}
\DeclareComplementaryOption{portrait}{landscape}
\DeclareStringOption[a4paper]{DIN}[a4paper]
% ------------------------------------------------------------------------------
% INCLUDE CONFIG FILE IF IT EXISTS  
% ------------------------------------------------------------------------------
% Margins	
%----------------------------------------------------------------------------------------
\InputIfFileExists{formularyETH/margins.cfg}{}{\PackageError{formulary}{"formularyETH/margins.cfg" does not exist}{}} 
\DeclareStringOption[\paperMarginTop]{top}[\paperMarginTop]
\DeclareStringOption[\paperMarginLeft]{left}[\paperMarginLeft]
\DeclareStringOption[\paperMarginRight]{right}[\paperMarginRight]
\DeclareStringOption[\paperMarginBottom]{bottom}[\paperMarginBottom]
\DeclareStringOption[\marginparwidth]{marginparwidth}[\marginparwidth]
\DeclareStringOption[\marginparsep]{marginparsep}[\marginparsep]
\DeclareStringOption[\headheight]{headheight}[\headheight]
\DeclareStringOption[\footskip]{footskip}[\footskip]
\DeclareStringOption[\headsep]{headsep}[\headsep]

% Number of columns per page 
% ----------------------------------------------------------------------
% Define default number of columns 
\newcounter{numberOfColumns}
\setcounter{numberOfColumns}{1}
\newbool{multicols}
\newif\ifsinglecolumn

\DeclareVoidOption{oneColumn}{\setcounter{numberOfColumns}{1} \singlecolumntrue}
\DeclareVoidOption{twoColumns}{\setcounter{numberOfColumns}{2} \multicolstrue \singlecolumnfalse}
\DeclareVoidOption{threeColumns}{\setcounter{numberOfColumns}{3} \multicolstrue\singlecolumnfalse}
\DeclareVoidOption{fourColumns}{\setcounter{numberOfColumns}{4} \multicolstrue\singlecolumnfalse\singlecolumnfalse}
\DeclareVoidOption{fiveColumns}{\setcounter{numberOfColumns}{5} \multicolstrue\singlecolumnfalse}
% Paper Orientation
% ----------------------------------------------------------------------
\ifmulticols\singlecolumnfalse\else\singlecolumntrue\fi
% Defaults
% 	\PassOptionsToClass{<options-list>} {<class-name>}:
%  	used to automatically invoke the corresponding options when the 
%	class is loaded.
%	\DeclareOption{<option>}{<code>}

% Colors	
%----------------------------------------------------------------------------------------
\RequirePackage[dvipsnames,table]{xcolor}
\InputIfFileExists{formularyETH/colors.cfg}{}{\PackageError{formulary}{"formularyETH/colors.cfg" does not exist}{}} 

% ==============================================================================
% Font Options
% ==============================================================================
\DeclareStringOption[10pt]{fontsize}[10pt]
%% 'sans serif' option
\DeclareVoidOption{sans}{\renewcommand{\familydefault}{\sfdefault}}
%% 'roman' option
\DeclareVoidOption{roman}{\renewcommand{\familydefault}{\rmdefault}} 
% Define the \theLanguage variable according to the chose option
% ----------------------------------------------------------------------
	\DeclareVoidOption{german}{\def\theLanguage{ngerman}}
	\DeclareVoidOption{english}{\def\theLanguage{english}}

% Code to be executed for unknown class option with name \CurrentOption
% ------------------------------------------------------------------------------ 
%----------------------------------------------------------------------------------------
% Packages	
%----------------------------------------------------------------------------------------
\ProcessKeyvalOptions* 
\PassOptionsToClass{\report@fontsize}{\report@baseClass}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{\report@baseClass}}
% Process given options 
% ------------------------------------------------------------------------------ 

% Loads article class that allows extra font sizes
\LoadClass{\report@baseClass} % Load the base class

%----------------------------------------------------------------------------------------
% PAPER GEOMETRY	
%----------------------------------------------------------------------------------------
\RequirePackage{geometry}
\geometry{
  \report@DIN,
  landscape=\boolean{report@landscape}, 
  top=\report@top, % Top margin
	bottom=\report@bottom, % Bottom margin
	left=\report@left, % Left margin
	right=\report@right, % Right margin
	headheight=\report@headheight, % Header height
	footskip=\report@footskip, % Space from the bottom margin to the baseline of the footer
	headsep=\report@headsep, % Space from the top margin to the baseline of the header
  marginparwidth=\report@marginparwidth,
  marginparsep=\report@marginparsep,
 	%showframe, % Uncomment to show how the type block is set on the page
}
%----------------------------------------------------------------------------------------
%	PACKAGES AND OTHER DOCUMENT CONFIGURATIONS
%----------------------------------------------------------------------------------------

	



% ======================================================================
% Packages
% ======================================================================
	% Basic packages formularyETH packages
	% load formularyETH packages
	\RequirePackage{formularyETH/formularyETH}			% Basic style definitions
  \InputIfFileExists{formularyETH/colors.cfg}{}{\PackageError{formulary}{"formularyETH/colors.cfg" does not exist}{}} 
	\RequirePackage{formularyETH/formularyETH_macros}	% Further macros
	\RequirePackage{formularyETH/formularyETH_boxes}	% section boxes
	\RequirePackage[utf8]{inputenc}	
	\RequirePackage{makeidx}							% Create an index
	\RequirePackage[super]{nth} 						% Allows to write \nth{1}, \nth{2},... for nicer formatting 
	%\DeclareDocumentCommand{\foocmd}{ O{default1} O{default2} m }{#1~#2~#3}
	%\foocmd[nondefault1]{foo}
	%\foocmd[nondefault2][nondefault1]{foo}
	\RequirePackage{import}								% Allows to specify import paths
	   	                    								% 
% ======================================================================
% Multicolumn Layout
% ======================================================================
\RequirePackage{multicol}			% for columns
\ifmulticols
	% start multicolumn
	% \AtBeginDocument{
	%     \begin{multicols*}{\value{numberOfColumns}}
	% }
	% Makes sure that columns are flusht to top other wise big spaces
	% \AtEndDocument{
	% 	\label{LastPage}
	% 	\end{multicols*}
	% }
	\setlength\columnsep{\columnsepperation}
\fi

% ======================================================================
% Basic Settings
% ======================================================================
% allows to append to graphicspath variable
\newcommand\appendtographicspath[1]{%
  \g@addto@macro\Ginput@path{#1}%
}

% ======================================================================
% Basic LaTeX Settings
% ======================================================================
  %\setlength{\parindent}{0mm} % Suppress paragraph indentation
	% \hyphenpenalty=10000			  % Almost no word breaks.
	% \pretolerance=60
	% \tolerance=2000
	% \doublehyphendemerits=10000       % No consecutive line hyphens.
	% \brokenpenalty=10000              % No broken words across columns/pages.
	% \widowpenalty=9999                % Almost no widows at bottom of page.
	% \clubpenalty=9999                 % Almost no orphans at top of page.
	% \interdisplaylinepenalty=100
	% \interfootnotelinepenalty=9999    % Almost never break footnotes.
	% \interlinepenalty=10000
	%\binoppenalty=200
	%\relpenalty=200