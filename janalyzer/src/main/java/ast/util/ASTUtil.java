package ast.util;

import ast.node.WrapedNode;
import com.alibaba.fastjson.JSONObject;
import com.alibaba.fastjson.serializer.SimplePropertyPreFilter;
import com.github.javaparser.ast.Node;

import java.util.ArrayList;
import java.util.List;

public class ASTUtil {
    public static String getTreeInJsonObject(Node node){
        if(node == null){
            return "[]";
        }
        WrapedNode wrapedNode = new WrapedNode(node);
        WrapedNode biNode = binarize(wrapedNode);
        SimplePropertyPreFilter filter = new SimplePropertyPreFilter();
        filter.getExcludes().add("node");
        return JSONObject.toJSONString(biNode, filter);
    }

    private static WrapedNode binarize(WrapedNode wrapedNode){

        WrapedNode biNode = new WrapedNode();
        List<WrapedNode> children = wrapedNode.getChildren();
        int child_num = children.size();
        if (child_num > 2){
            biNode.setMask(wrapedNode.getMask());
            biNode.setNodeName(wrapedNode.getNodeName());
            WrapedNode left = binarize(children.get(0));
            WrapedNode newRight = new WrapedNode("binaryOperator","0");
            for (int i = 1; i < child_num; i++){
                newRight.setChild(children.get(i));
            }

            biNode.setChild(left);
            biNode.setChild(binarize(newRight));
        }else if(child_num == 2){
            biNode.setMask(wrapedNode.getMask());
            biNode.setNodeName(wrapedNode.getNodeName());
            WrapedNode left = binarize(children.get(0));
            WrapedNode right = binarize(children.get(1));
            biNode.setChild(left);
            biNode.setChild(right);
        }else if (child_num == 1){
            WrapedNode child = children.get(0);
            biNode = binarize(child);
            String mask = biNode.getMask();
            String nodeName = wrapedNode.getNodeName() + "." + biNode.getNodeName();
            biNode.setNodeName(nodeName);
            biNode.setMask(mask);
        }else{
            biNode.setMask(wrapedNode.getMask());
            biNode.setNodeName(wrapedNode.getNodeName());
        }
        return biNode;
    }
}
