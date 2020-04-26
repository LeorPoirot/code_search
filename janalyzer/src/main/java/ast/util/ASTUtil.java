package ast.util;

import ast.node.WrapedNode;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SimplePropertyPreFilter;
import com.github.javaparser.ast.Node;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ASTUtil {
    public static String getTreeInJsonObject(Node node){
        if(node == null){
            return "[]";
        }
        WrapedNode wrapedNode = new WrapedNode(node);
        WrapedNode biNode = binarize(wrapedNode, 0, 10);
        SimplePropertyPreFilter filter = new SimplePropertyPreFilter();
        filter.getExcludes().add("node");
        return JSONObject.toJSONString(biNode, filter);
    }

    private static WrapedNode binarize(WrapedNode wrapedNode, int deepth, int max){

        WrapedNode biNode = new WrapedNode();
        List<WrapedNode> children = wrapedNode.getChildren();
        int child_num = children.size();
        if (deepth == max){
            biNode.setChild(null);
            biNode.setMask("1");
            biNode.setNodeName(wrapedNode.getNodeName());
            return biNode;
        }

        if (child_num > 2){
            biNode.setMask(wrapedNode.getMask());
            biNode.setNodeName(wrapedNode.getNodeName());
            WrapedNode left = binarize(children.get(0), ++deepth, max);
            WrapedNode newRight = new WrapedNode("binaryOperator","0");
            for (int i = 1; i < child_num; i++){
                newRight.setChild(children.get(i));
            }

            biNode.setChild(left);
            biNode.setChild(binarize(newRight, ++deepth, max));
        }else if(child_num == 2){
            biNode.setMask(wrapedNode.getMask());
            biNode.setNodeName(wrapedNode.getNodeName());
            WrapedNode left = binarize(children.get(0), ++deepth, max);
            WrapedNode right = binarize(children.get(1), ++deepth, max);
            biNode.setChild(left);
            biNode.setChild(right);
        }else if (child_num == 1){
            WrapedNode child = children.get(0);
            child.setNodeName(wrapedNode.getNodeName() + "." + child.getNodeName());
            biNode = binarize(child, ++deepth, max);
        }else{
            biNode.setMask(wrapedNode.getMask());
            String nodeName = wrapedNode.getNodeName();
            String rex = "([_\\d\\w]+\\.)*[_\\d\\w]+$";
            Pattern p = Pattern.compile(rex);
            Matcher m = p.matcher(nodeName);
            Boolean is_fined = m.lookingAt();
            if (!is_fined){
                biNode.setNodeName("<blank>");
            }else {
                biNode.setNodeName(nodeName);
            }
        }
        return biNode;
    }
}
