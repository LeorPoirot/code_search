package project;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;


public class JOutput {
    private File repo_file;
    // description
    private File fileComment;
    private File fileJavadoc;
    // raw code
    private File fileSourceCode;
    // API sequence
    private File fileParsedCode;
    // method name, parameters, returns
    private File fileMethod;
    // parameter
    private File fileParameter;
    // return
    private File fileReturn;
    private File fileModifiers;

    private File fileSource;
    private File fileAst;
    private File fileHash;

    private String projectName;
    private String fileName;




    public int cCmt = 0;
    public int cDoc = 0;
    public int cSrc = 0;
    public int cPrs = 0;
    public int cMet = 0;
    public int cPar = 0;
    public int cRet = 0;
    public int cMod = 0;
    public int cSce = 0;
    public int cAst = 0;
    public int cHash = 0;

    private File fileJDK;
    private HashMap<String, String> jdk;

    public JOutput(String path) {
        fileJDK = new File(path);
        try {
            if (fileJDK.exists()) {
                System.out.println(fileJDK.delete());
            }
            System.out.println(fileJDK.createNewFile());
        } catch (Exception e) {
            System.out.println(e);
        }
    }

    public JOutput(String path, String projectName, HashMap<String, String> jdk) {
        this.jdk = jdk;
        this.projectName = projectName;
        // String i = projectName.substring(0, projectName.length() - 4);
        String i = projectName;
        repo_file = new File(path + this.projectName);
        fileComment = new File(path + this.projectName + "/file_" + i + "_Comment.csv");
        fileJavadoc = new File(path + this.projectName + "/file_" + i + "_Javadoc.csv");
        fileSourceCode = new File(path + this.projectName + "/file_" + i + "_SourceCode.csv");
        fileSource = new File(path + this.projectName + "/file_" + i + "_Source.csv");
        fileParsedCode = new File(path + this.projectName + "/file_" + i + "_ParsedCode.csv");
        fileMethod = new File(path + this.projectName + "/file_" + i + "_Method.csv");
        fileParameter = new File(path + this.projectName + "/file_" + i + "_Parameter.csv");
        fileReturn = new File(path + this.projectName + "/file_" + i + "_Return.csv");
        fileModifiers = new File(path + this.projectName + "/file_" + i + "_Modifiers.csv");
        fileAst = new File(path + this.projectName + "/file_" + i + "_Ast.csv");
        //fileHash = new File(path + this.projectName + "\\file_" + i + "_Hash.csv");
        if (!repo_file.exists()){
            repo_file.mkdir();
        }
        clear();
        build();
    }

    public void clear(){
        try {
            if (fileComment.exists()) {
                fileComment.delete();
            }
            if (fileJavadoc.exists()) {
                fileJavadoc.delete();
            }
            if (fileSourceCode.exists()) {
                fileSourceCode.delete();
            }
            if (fileSource.exists()) {
                fileSource.delete();
            }
            if (fileParsedCode.exists()) {
                fileParsedCode.delete();
            }
            if (fileMethod.exists()) {
                fileMethod.delete();
            }
            if (fileParameter.exists()) {
                fileParameter.delete();
            }
            if (fileReturn.exists()) {
                fileReturn.delete();
            }
            if (fileModifiers.exists()) {
               fileModifiers.delete();
            }
//            if (fileHash.exists()){
//                fileHash.delete();
//            }
            if (fileAst.exists()){
                fileAst.delete();
            }
        }catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public void build(){
        try {
            fileComment.createNewFile();
            fileJavadoc.createNewFile();
            fileSourceCode.createNewFile();
            fileSource.createNewFile();
            fileParsedCode.createNewFile();
            fileMethod.createNewFile();
            fileParameter.createNewFile();
            fileModifiers.createNewFile();
            fileReturn.createNewFile();
            fileAst.createNewFile();
            //fileHash.createNewFile();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public void write(File file, String line) {
        try {
            FileOutputStream fos = new FileOutputStream(file, true);
            OutputStreamWriter osw = new OutputStreamWriter(fos, StandardCharsets.UTF_8);
            BufferedWriter bw = new BufferedWriter(osw);

            if (line == null) {
                return;
            }
            bw.write(line + "\r\n");

            bw.flush();
            bw.close();
            osw.close();
            fos.close();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public void write(File file, String line, int idx) {
        try {
            FileOutputStream fos = new FileOutputStream(file, true);
            OutputStreamWriter osw = new OutputStreamWriter(fos, StandardCharsets.UTF_8);
            BufferedWriter bw = new BufferedWriter(osw);

            bw.write(idx + ";");
            if (line == null) {
                line = "[]";
            }
            bw.write(line + "\r\n");

            bw.flush();
            bw.close();
            osw.close();
            fos.close();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public void write(File file, List<String> lines, String delimiter, int idx) {
        try {
            FileOutputStream fos = new FileOutputStream(file, true);
            OutputStreamWriter osw = new OutputStreamWriter(fos, StandardCharsets.UTF_8);
            BufferedWriter bw = new BufferedWriter(osw);

            bw.write(idx + ";");
            if (lines == null) {
                bw.write("[]\r\n");
            } else {
                if (lines.isEmpty()) {
                    bw.write("[]" + "\r\n");
                } else {
                    for (String line : lines) {
                        bw.write(line + delimiter);
                    }
                    bw.write("\r\n");
                }
            }

            bw.flush();
            bw.close();
            osw.close();
            fos.close();
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public File getFileComment() {
        return fileComment;
    }

    public void setFileComment(File fileComment) {
        this.fileComment = fileComment;
    }

    public File getFileJavadoc() {
        return fileJavadoc;
    }

    public void setFileJavadoc(File fileJavadoc) {
        this.fileJavadoc = fileJavadoc;
    }

    public File getFileSourceCode() {
        return fileSourceCode;
    }

    public void setFileSourceCode(File fileSourceCode) {
        this.fileSourceCode = fileSourceCode;
    }

    public File getFileParsedCode() {
        return fileParsedCode;
    }

    public void setFileParsedCode(File fileParsedCode) {
        this.fileParsedCode = fileParsedCode;
    }

    public File getFileMethod() {
        return fileMethod;
    }

    public void setFileMethod(File fileMethod) {
        this.fileMethod = fileMethod;
    }

    public String getProjectName() {
        return projectName;
    }

    public void setProjectName(String projectName) {
        this.projectName = projectName;
    }

    public String getFileName() {
        return fileName;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public File getFileParameter() {
        return fileParameter;
    }

    public void setFileParameter(File fileParameter) {
        this.fileParameter = fileParameter;
    }

    public File getFileReturn() {
        return fileReturn;
    }

    public void setFileReturn(File fileReturn) {
        this.fileReturn = fileReturn;
    }

    public File getFileJDK() {
        return fileJDK;
    }

    public void setFileJDK(File fileJDK) {
        this.fileJDK = fileJDK;
    }

    public File getFileModifiers() {
        return fileModifiers;
    }

    public void setFileModifiers(File fileModifiers) {
        this.fileModifiers = fileModifiers;
    }

    public HashMap<String, String> getJdk() {
        return jdk;
    }

    public void setJdk(HashMap<String, String> jdk) {
        this.jdk = jdk;
    }

    public File getFileSource() {
        return fileSource;
    }

    public void setFileSource(File fileSource) {
        this.fileSource = fileSource;
    }

    public File getFileAst() {
        return fileAst;
    }

    public void setFileAst(File fileAst) {
        this.fileAst = fileAst;
    }

    public File getFileHash() {
        return fileHash;
    }

    public void setFileHash(File fileHash) {
        this.fileHash = fileHash;
    }

}
