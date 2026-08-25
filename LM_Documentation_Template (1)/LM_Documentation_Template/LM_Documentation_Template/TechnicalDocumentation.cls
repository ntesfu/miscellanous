\NeedsTeXFormat{LaTeX2e} % Defines the LaTeX version to use
\ProvidesClass{TechnicalDocumentation} % This file outputs "MyClass" to be used in other documents


% Process options passed to the class
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}% This command loads all the options for the memoir class (12pt, article, twosided, etc.)
\ProcessOptions\relax
\LoadClass{memoir} % Load the memoir class. 


% Packages that I usually use in my reports and documents
\RequirePackage{amsmath}
\RequirePackage{amsfonts}
\RequirePackage{datetime}
\RequirePackage{amssymb}
\RequirePackage{pdfpages}
\RequirePackage{graphicx}
\RequirePackage{float}
\RequirePackage{subcaption}
\RequirePackage{wrapfig}
\RequirePackage{fancyhdr}
\RequirePackage{textcomp}
\RequirePackage{xcolor}
\RequirePackage{makecell} % Provides extra powerful features for multi-line cells in tables
\RequirePackage[explicit]{titlesec}
\RequirePackage{pdflscape}
\RequirePackage{longtable}
\RequirePackage{lipsum}
\RequirePackage{multirow}
\RequirePackage{epstopdf}
\RequirePackage{multicol}
\RequirePackage{xargs}
\RequirePackage{tabularx}
\RequirePackage{booktabs}
\RequirePackage{cprotect} % Add verbatium everywhere including chapters and sections
\RequirePackage{hyperref}
\RequirePackage{xifthen} % If-else programming to LaTeX. Used to check for empty arguments in some functions
\DisemulatePackage{setspace}
\RequirePackage{setspace}

\RequirePackage[most]{tcolorbox}
\tcbuselibrary{listings}
\usepackage{algorithm}
\usepackage{algpseudocode}
% \usepackage{algorithmic}
% \usepackage[table,xcdraw]{xcolor}

\usepackage[width=17.5cm, height=26cm]{geometry} % paper dimensions


% Document encoding, font, and language
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}

% Set the depth of table of contents to subsections. This will allow
% subsections to be properly numbered and be visible in the ToC.
\setsecnumdepth{subsubsection}
\settocdepth{subsection}

% Coloring for chapters, sections, subsections, subsubsections, and code listings 
\definecolor{darkBlue}{RGB}{26,54,104}


% default graphics paths
\graphicspath{{pic/}{assets/}}

% define the cover page
\newcommand{\coverpage}[5]{
	\begin{titlingpage}
		\centering

		\ifthenelse{\isempty{#5}}{}{
			\begin{figure}[H]
				\centering
				\includegraphics[width=0.5\textwidth]{#5}
			\end{figure}
		}

		\vspace*{10pt}
		\begin{center}
			\begin{vplace}[0.7]
				Center for Innovation and Security Solutions\\
				#3
				\begin{tcolorbox}[
						spread sidewards,
						colframe=darkBlue,
						colback=white,
						coltext=black,
						boxrule=3pt,
						leftrule=0mm,
						rightrule=0mm,
						boxsep=4mm,
						arc=0mm,
						halign=center,
					]
					\LARGE{\textbf{#1}}
				\end{tcolorbox}
				\bigskip
				by\\[10pt]
				{	\hypersetup{hidelinks}
				#2 %author setup to hide links 
				}

			\end{vplace}
			\begin{center}
				#4\\
				\the\year
			\end{center}

		\end{center}
	\end{titlingpage}
}

\definecolor{codebg}{RGB}{253, 246, 227}
\definecolor{codefg}{RGB}{101, 123, 131}
\definecolor{codegreen}{RGB}{133, 153, 0}
\definecolor{codegray}{RGB}{147, 161, 161}
\definecolor{codecyan}{RGB}{42, 161, 152}
\definecolor{codepurple}{RGB}{139, 0, 139}
% \definecolor{darkBlue}{RGB}{42, 58, 98}
% \tcbuselibrary{minted,skins}
% Define code listing box
\usepackage[ttdefault=false]{AnonymousPro} % font for code lists
\newtcblisting[auto counter,number within=chapter]{listing}[3][]{
	enhanced,
	breakable,
	colback=darkBlue!10!white,
	% title=#1,
	title=Listing ~\thetcbcounter: #2,
	attach boxed title to top right={yshift=-\tcboxedtitleheight},
	fonttitle=\color{black}\itshape\ttfamily,
	boxed title style={
			size=small,
			colframe=darkBlue!10!white,
			colback=darkBlue!20!white,
			%sharp corners=uphill,
			%arc=.5cm,
			top=1mm,bottom=1mm,left=1mm,right=1mm
		},
	frame empty,
	boxrule=0pt,
	left=0pt,
	top=0pt,
	bottom=0pt,
	right=0pt,
	listing only,
	listing options={
			basicstyle=\color{black}\fontfamily{AnonymousPro}\selectfont\small,
			language={#3},
			showspaces=false,
			showtabs=false,
			breaklines=true,
			showstringspaces=false,
			tabsize=4,
			upquote=true,
			escapeinside={(*@}{@*)},
			commentstyle=\color{codegray},
			keywordstyle=\color{codegreen},
			stringstyle=\color{codecyan},
			identifierstyle=\color{darkBlue}
		},
	#1
}




% Define terminal box 
\newtcblisting{terminal}{
	enhanced,
	colback=black,
	attach boxed title to top right={yshift=-\tcboxedtitleheight},
	frame empty,
	boxrule=0pt,
	left=0pt,
	top=0pt,
	bottom=0pt,
	right=0pt,
	listing only,
	listing options={
			basicstyle=\color{white}\fontfamily{AnonymousPro}\selectfont\scriptsize,
			showspaces=false,
			showtabs=false,
			breaklines=true,
			showstringspaces=false,
			tabsize=4,
			upquote=true,
			escapeinside={(*@}{@*)}

		},
}

% Define warning box
\newtcolorbox{warning}[1][]{enhanced,
	before skip=2mm,after skip=3mm,
	boxrule=0pt,left=4mm,right=2mm,top=1mm,bottom=1mm,
	colback=yellow!30!white,
	colframe=yellow!20!black,
	sharp corners,
	underlay={%
			\path[fill=yellow!30!black,draw=none] (interior.south west) rectangle node[white]{\Huge\bfseries !} ([xshift=3mm]interior.north west);
		},
	drop fuzzy shadow,#1
}

% Define a tip box
\definecolor{beige}{RGB}{227,244,227}
\definecolor{mediumseagreen}{RGB}{84,180,73}
\newtcolorbox{tip}{
	enhanced,
	breakable,
	title=TIP,
	fonttitle=\sffamily\bfseries,
	coltitle=beige,
	attach boxed title to top right={yshift=-3mm,yshifttext=-1mm,xshift=-5mm},
	boxed title style={size=small,colback=mediumseagreen,sharp corners},
	colback=beige,
	colframe=mediumseagreen,
	boxrule=3mm,rightrule=0mm,toprule=0mm, bottomrule=0mm,
	sharp corners,
	drop fuzzy shadow
}

% Format a numbered chapter
\titleformat{\chapter}[display]{\bfseries\Large}{}{0pt}{
	\begin{tcolorbox}[
			spread sidewards,
			colback=darkBlue,
			colframe=darkBlue!40!white,
			coltext=white,
			boxrule=5pt,
			leftrule=0mm,
			rightrule=0mm,
			boxsep=4mm,
			arc=0mm,
			%		left=0.1in+\hoffset+\oddsidemargin,
			%		right=\paperwidth-0.1in-\hoffset-\oddsidemargin-\textwidth,
			after skip=-0.5cm,
		]
		\Large\chaptertitlename\enspace\thechapter:\quad #1
	\end{tcolorbox}
}

% Format number-less chapter
\titleformat{name=\chapter,numberless}[display]{\bfseries\Large}{}{0cm}{
	\begin{tcolorbox}[
			spread sidewards,
			colback=darkBlue,
			colframe=darkBlue!40!white,
			coltext=white,
			boxrule=5pt,
			leftrule=0mm,
			rightrule=0mm,
			boxsep=4mm,
			arc=0mm,
			%		left=0.1in+\hoffset+\oddsidemargin,
			%		right=\paperwidth-0.1in-\hoffset-\oddsidemargin-\textwidth,
			after skip=-0.5cm,
		]
		#1
	\end{tcolorbox}
}
\titlespacing*{\chapter}{0pt}{-0.5cm}{0cm}


% Format a numbered section
\newcommand{\sectionAfterSkip}{-0.65cm} % spacing for after skip
\newcommand{\sectionBeforeSkip}{-0.65cm} % spacing for before skip
\titleformat{\section}[display]{\bfseries\Large}{}{0pt}{
	\begin{tcolorbox}[
			colback=white,
			colframe=darkBlue,
			coltext=darkBlue,
			boxrule=5pt,
			leftrule=0mm,
			rightrule=0mm,
			toprule=0mm,
			arc=0mm,
			left=-1.2mm,
			after skip=0cm,
			before skip=0cm
		]
		\Large\thesection\quad #1
	\end{tcolorbox}
}

% Format number-less section
\titleformat{name=\section,numberless}[display]{\bfseries\Large}{}{0cm}{
	\begin{tcolorbox}[
			colback=white,
			colframe=darkBlue,
			coltext=darkBlue,
			boxrule=5pt,
			leftrule=0mm,
			rightrule=0mm,
			toprule=0mm,
			arc=0mm,
			left=-1.2mm,
		]
		#1
	\end{tcolorbox}
}
\titlespacing*{\section}{0cm}{\sectionBeforeSkip}{\sectionAfterSkip}

% Format a numbered subection
\newcommand{\subsectionAfterSkip}{-0.5cm} % spacing for after skip
\newcommand{\subsectionBeforeSkip}{-0.5cm} % spacing for before skip
\titleformat{\subsection}[display]{\bfseries\large}{}{0pt}{
	\begin{tcolorbox}[
			colback=white,
			colframe=darkBlue,
			coltext=darkBlue,
			boxrule=1pt,
			leftrule=0mm,
			rightrule=0mm,
			toprule=0mm,
			arc=0mm,
			left=-1.2mm,
			after skip=0cm,
			before skip=0cm
		]
		\large\thesubsection\quad #1
	\end{tcolorbox}
}

% Format number-less subection
\titleformat{name=\subsection,numberless}[display]{\bfseries\large}{}{0cm}{
	\begin{tcolorbox}[
			colback=white,
			colframe=darkBlue,
			coltext=darkBlue,
			boxrule=1pt,
			leftrule=0mm,
			rightrule=0mm,
			toprule=0mm,
			arc=0mm,
			left=-1.2mm,
			after skip=0cm,
			before skip=0cm
		]
		#1
	\end{tcolorbox}
}
\titlespacing*{\subsection}{0pt}{\sectionBeforeSkip}{\subsectionAfterSkip}


% Format a numbered subsubection
\newcommand{\subsubsectionAfterSkip}{\baselineskip} % spacing for after skip
\newcommand{\subsubsectionBeforeSkip}{-0.1cm} % spacing for before skip
\titleformat{\subsubsection}[display]{\bfseries}{}{0pt}{
	\textcolor{darkBlue}{\thesubsubsection\quad #1}
}

% Format number-less subection
\titleformat{name=\subsubsection,numberless}[display]{\bfseries}{}{0cm}{
	\textcolor{darkBlue}{#1}
}

\titlespacing*{\subsubsection}{0pt}{\subsubsectionBeforeSkip}{\subsubsectionAfterSkip}



% Table of contents without hyperref colors
\newcommand{\toc}{
	{
			\hypersetup{hidelinks}
			\tableofcontents
		}
}

% List of figures without hyperref colors
\newcommand{\lof}{
	{
			\hypersetup{hidelinks}
			\listoffigures
		}
}

% List of tables without hyperref colors
\newcommand{\lot}{
	{
			\hypersetup{hidelinks}
			\listoftables
		}
}






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ZOOMBOX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newsavebox\zbox
\newcounter{zoom}
\newcommand{\zoombox}[2][0]{%
\leavevmode%
\sbox\zbox{#2}%
\pdfdest name {zb\thezoom.out} fit
\pdfdest name {zb\thezoom.in} fitr
width  \the\wd\zbox\space
height \the\ht\zbox\space
depth  \the\dp\zbox\space
\immediate\pdfannot
width  \the\wd\zbox\space
height \the\ht\zbox\space
depth  \the\dp\zbox\space
{%
/Subtype/Link/H/N
/Border [0 0 #1 [1 2]]
/A <<
/S/JavaScript
/JS (
if(typeof(zoomed\thezoom)=='undefined'){
		this.gotoNamedDest('zb\thezoom.out');
		var zoomed\thezoom=false;
	}
)
/Next <<
/S/GoTo/D (zb\thezoom.in)
/Next <<
/S/JavaScript
/JS(
if(zoomed\thezoom){
		this.gotoNamedDest('zb\thezoom.out');
		zoomed\thezoom=false;
	}else{
zoomed\thezoom=true;
}
)
>>
>>
>>
}%
\usebox{\zbox}%
\stepcounter{zoom}%
}
\makeatother
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ZOOMBOX %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
