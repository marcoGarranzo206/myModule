import subprocess
from time import time
import os

def chemTok(sentence, path ,direc = "/tmp/",remove_file = True):

    """
    tokenize sentence (a string) using chemtok
    LINK

    can handle multiple sentences if they are separated by newline
    chemtok returns as output a token per line

    returns a list of lists. each list is a sentence, each element inside
    a token
    """

    file_name = direc + f"chemtoInput_{time()}.txt"
    with open(file_name, "w+") as f:

        f.write(sentence)

    with open(file_name, "r") as f:

        command = ["java", "-jar", path + "chemtok-1.0.1.jar"]
        #print(command)
        result = subprocess.run(command,stdin = f, capture_output=True)

    if remove_file:

        os.remove(file_name)

    if result.stderr:

        print(result.stderr.decode())
        raise subprocess.CalledProcessError(result.returncode, cmd = command, stderr = result.stderr)


    #if multiple sentences, they will be separated by two newlines
    #split by sentence
    result_list = result.stdout.decode().strip().split("\n\n")
    #inside each sentence, split by token
    result_list = [x.split("\n") for x in result_list]

    return result_list

def bioLemmatizer(list_of_tokens, path,direc = "/tmp/",remove_files = True):

    """
    use biolemmatizer to lemmatize a list of tokens with
    optionally their POS

    list_of_tokens: a list of lists

    each list: a sentence
    each element: either a string (token) or tuple of (token,POS)

    returns: list of lists

    each list a sentence
    each element its lemma

    Uses unbuilt biolemmatizer, cannot get the permanent build to work in my machine
    as such, better to lemmatize by batch, to avoid the momentary building
    it needs to do every ttime to execute it
    """

    sessionID = time()

    input_file_name = direc + f"bioLemmatizerInput_{sessionID}.txt"
    output_file_name = direc + f"bioLemmatizerOutput_{sessionID}.txt"
    command_file = direc + f"commandBioLemmatizer_{sessionID}.sh"
    logFile = direc + f"bioLemmatizerLog_{sessionID}.txt"

    has_POS = len(list_of_tokens[0][0]) == 2

    try:

        with open(input_file_name, "w+") as f:

            for i,sentence in enumerate(list_of_tokens):

                if has_POS:

                    for token,POS in sentence:

                        f.write(token+"\t"+POS+"\n")
                else:

                    for token in sentence:

                        f.write(token+"\n")

                if i < len(list_of_tokens) - 1:

                    f.write("\n")

        # a bit of an ugly work-around
        # but I just cant get the command working through subprocess module
        # problems with edu.ucdenver.ccp.nlp.biolemmatizer.BioLemmatizer not being found
        # however, if written to bash file and execute, it works

        with open(command_file,"w+") as f:

            f.write(f'mvn -X -f {path}biolemmatizer-core/pom.xml exec:java\
            -Dexec.mainClass="edu.ucdenver.ccp.nlp.biolemmatizer.BioLemmatizer"\
            -Dexec.args="-i {input_file_name} -o {output_file_name} -l"')

        result = subprocess.run(["bash", command_file], capture_output=True)

        with open(logFile, "w+") as f:

            f.write(result.stdout.decode())

        if result.stderr:

            print(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, cmd = ["bash", command_file], stderr = result.stderr, shell = True)

        #if multiple sentences, they will be separated by two newlines
        #split by sentence

        output = []
        col = 2 if has_POS else 1
        with open(output_file_name,"r") as f:

            output.append([])
            for line in f:

                #print(line)
                if line == "\n":

                    output.append([])

                else:

                    output[-1].append(line[:-1].split("\t")[col].split("||"))
    finally:

        if remove_files:
            #remove in order of creation
            #if one hasnt been created and error
            #all create will be removed
            #but it will throw an error
            os.remove(input_file_name)
            os.remove(output_file_name)
            os.remove(command_file)
            os.remove(logFile)


    return output
