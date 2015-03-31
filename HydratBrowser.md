# Introduction #

The hydrat browser can help users read, organise and display the data in the data store (.h5 file) via a internet browser


# Details #
## Basic Usage ##
```
hydrat browse [options] datafile
```

## Parameters ##
  * **-m**: view the data store in modifiable manner. This parameter also enables the hydrat to update the old store to the newest version.

  * **-b**: open the view in a new tab/browser

  * **-r**: enable remote access to the data store. By default, the address is 127.0.0.1:8080.

  * **-p**: change the proxy. By default, the proxy is 8080

## Other tips ##
  * To configure what to display in the browser, you can change the code in browser\_config.py in the current directory.

  * To generate a summary of the results before viewing via _hydrat browse_:
```
hydrat summary datafile
```

  * To generate the old view of the results summary:
```
hydrat output datafile /path/to/folder
```

> It is also possible to upload the files when generating the the summary:
```
hydrat output datafile ssh:username@some.server.com
```
