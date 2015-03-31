# NOTE #

**This document is under development. It talks about an incomplete feature of hydrat, and some of the ideas it describes may not yet have been implemented!**

# Introduction #

One of hydrat's strengths is that it tries to avoid duplicating others' work. Where possible we have made an effort to provide standardized interfaces to a wide variety of toolkits. Examples of this include:

  * TODO

One drawback of this is that in order to be usable, each toolkit must be installed by the user, and hydrat must be informed of where to find it. hydrat does not provide much assistance with the former task, as it is much better handled by the myriad of package managers. Admittedly, most of the tools that hydrat uses are not packaged by any standardized distribution method, but that is a different problem altogether. However, hydrat does provide assistance with locating installed tools, via its configuration system, described in this document.

# User Interface #

  * Describe the interface. User just calls hydrat configure 

&lt;file&gt;



# Programmer API #

  * Describe what a programmer needs to do to hook into hydrat's configuration functionality.

# How it works #

  * TODO: Figure out how to get hydrat to scan every object for configurability. This is also something we want to do only when configure is called, not at every import of hydrat.

# TODO #
  * Make the tool update an existing configuration if it already exists
  * Perhaps like the linux kernel's make oldconfig, interactively query the user about config options to be changed. Have flags for interactive mode (-i) and auto-override (-o).