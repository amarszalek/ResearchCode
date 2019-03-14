%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% IPM2019 class for LaTeX files                v1.0   25-10-2018  %%
%%                                                                 %%
%% The regular article class commands still exist and may be used. %% 
%%                                                                 %%
%% \maketitle makes the first page automatically     							 %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{ShortPaper_Instructions_LaTeX_IPM2019}

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\DeclareOption{dvips}{%
  \PassOptionsToPackage{dvips}{hyperref}}
\DeclareOption{pdftex}{%
  \PassOptionsToPackage{pdftex}{hyperref}}
\DeclareOption{dvipdfm}{%
  \PassOptionsToPackage{dvipdfm}{hyperref}}
\DeclareOption{dvips}{%
  \PassOptionsToPackage{dvips}{graphicx}}
\DeclareOption{pdftex}{%
  \PassOptionsToPackage{pdftex}{graphicx}}
\DeclareOption{dvipdfm}{%
  \PassOptionsToPackage{dvipdfm}{graphicx}}

\ProcessOptions\relax

\LoadClass[a4paper,10pt,notitlepage,oneside]{article}

\RequirePackage{hyperref}
\RequirePackage{calc}
\RequirePackage{indentfirst}
\RequirePackage{url}
\RequirePackage{graphicx}
\RequirePackage{tgtermes}
%\RequirePackage{qtimes}
\RequirePackage{caption}
\RequirePackage{cite}
\RequirePackage{fancyhdr}

\hyphenpenalty10000
\exhyphenpenalty10000

%% Title
\renewcommand{\maketitle}{%
 \newpage
 %\thispagestyle{plain}
 \raisebox{8.2mm}[0mm][10mm]{
 \begin{minipage}[t][0mm][t]{124mm}
	\begin{flushright}
		\fontsize{9}{10}\selectfont
   			5$^{th}$ ECCOMAS Thematic Conference\\
   			on Inverse Problems Methods \textbf{IPM 2019}\\
   			22$^{nd}$  --24$^{th}$ May 2019, Rzesz\'{o}w--Kombornia, Poland
  	\end{flushright}
 \end{minipage}
 }		
 \begin{center}%
  \vspace{2pt}
  {%3
   \fontsize{12}{14}\selectfont
   \bfseries
   \MakeUppercase{\@title}
   \par
  }%3
  \vspace{12pt}%
  {%4
\fontsize{10}{12}\selectfont
   \bfseries\@author
   \par
  }%4
  \vspace{12pt}%
  {%5
\fontsize{9}{10}\selectfont
   \def\and{\\\vspace{12pt}}
   \normalfont\@address
   %\par
  }%5
 \end{center}
 %\vspace{12pt}
 %\vfill
 }% End of \@maketitle
 
\def\author{\@ifnextchar[{\@getshortauthor}{\@getauthor}}
\def\@getauthor#1{\gdef\@author{#1}\gdef\@shortauthor{#1}}
\def\@getshortauthor[#1]#2{\gdef\@shortauthor{#1}\gdef\@author{#2}}

\newcommand{\address}[1]{\gdef\@address{#1}}
%\newcommand{\keywords}[1]{\gdef\@keywords{#1}}

\renewcommand{\abstract}[1]{\gdef\@abstract{#1}}
%\newcommand{\heading}[1]{\gdef\@heading{#1}}

%% Header and footnote definition
\def\ps@myheadings{%
    %\def\@oddfoot{\normalfont\hfil\thepage\hfil}
    \let\@oddfoot\@empty	
    \let\@evenfoot\@oddfoot
%    \def\@evenhead{%
%        \parbox[t][10pt][t]{\textwidth}{%
%        %\centering
%        \normalfont
%        \fontsize{10}{5}\selectfont
%        \@heading\\
%%				\vspace{-.5cm}
%        \noindent\rule{\textwidth}{0.5pt}%
%     }
%    }%
%    \let\@oddhead\@evenhead
%    \let\@mkboth\@gobbletwo
%    \let\sectionmark\@gobble
%    \let\subsectionmark\@gobble
}

\def\sectionLevel{1}
\def\sectionIndent{0pt}
\def\sectionSpaceBefore{10pt}
\def\sectionSpaceAfter{4pt}
\def\sectionStyle{\normalsize\bfseries\MakeUppercase}

\renewcommand{\section}{%
  \@startsection{section}{\sectionLevel}{\sectionIndent}{\sectionSpaceBefore}%
                {\sectionSpaceAfter}{\sectionStyle}}%

\def\sectionLevel{2}
\def\sectionIndent{0pt}
\def\sectionSpaceBefore{10pt}
\def\sectionSpaceAfter{3pt}
\def\sectionStyle{\normalsize\bfseries}

\renewcommand{\subsection}{%
  \@startsection{subsection}{\sectionLevel}{\sectionIndent}{\sectionSpaceBefore}%
                {\sectionSpaceAfter}{\sectionStyle}}%

\renewcommand{\@makecaption}[2]{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{\normalsize\fontsize{10}{12}\selectfont #1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    \normalsize\fontsize{10}{12}\selectfont #1: #2\par
  \else
    \global \@minipagefalse
    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}

\renewenvironment{thebibliography}[1]
  {\smallskip
  \section*{\MakeUppercase\refname\@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}}%
  \list{\@biblabel{\@arabic\c@enumiv}}%
        {\settowidth\labelwidth{\@biblabel{#1}}%
         \leftmargin\labelwidth
         \topsep 0pt
	    	 \parsep 0pt
	    	 \partopsep 0pt
				 \itemsep \topsep
         \advance\leftmargin\labelsep
         \@openbib@code
         \usecounter{enumiv}%
         \let\p@enumiv\@empty
         \renewcommand\theenumiv{\@arabic\c@enumiv}}%
   \sloppy
   \clubpenalty4000
   \@clubpenalty \clubpenalty
   \widowpenalty4000%
   \sfcode`\.\@m}
  {\def\@noitemerr
   {\@latex@warning{Empty `thebibliography' environment}}%
   \endlist}
      
%% Page settings  
\setlength{\hoffset}{-3mm}
\setlength{\evensidemargin}{12.5mm}
\setlength{\voffset}{9mm}
\setlength{\headheight}{10pt}
\setlength{\headsep}{14pt}
\setlength{\topmargin}{15mm}
\setlength{\textheight}{188mm}
\setlength{\textwidth}{130mm}
\setlength{\textfloatsep}{6pt}
\setlength{\intextsep}{6pt} 
\setlength{\floatsep}{10pt}
\setlength{\abovedisplayskip}{0pt} 
\setlength{\belowdisplayskip}{0pt} 
\setlength{\parindent}{5mm}
%\setlength{\abovecaptionskip}{6pt}
%\setlength{\belowcaptionskip}{6pt}
\setlength{\itemsep}{0pt}


%\captionsetup[table]{font={footnotesize},justification=centerlast,singlelinecheck=yes,labelfont=bf,aboveskip=0pt,belowskip=0.7\baselineskip}
%\captionsetup[figure]{font={footnotesize},justification=raggedright,singlelinecheck=yes,labelfont=bf,aboveskip=0.7\baselineskip,belowskip=0pt}

\captionsetup[table]{justification=centerlast,singlelinecheck=yes,aboveskip=0pt,belowskip=6pt}
\captionsetup[figure]{singlelinecheck=yes,aboveskip=6pt,belowskip=-6pt}

\AtBeginDocument{%
  \pagestyle{myheadings}
  \maketitle
  \let\maketitle\relax
}

\endinput
