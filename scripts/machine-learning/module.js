import { ReverseFunction } from "./autodiff/reverse-function.js";
import { LinearLayer } from "./functions/layer/linear.js";


export class Module {
    
    /**
     * 
     * @param {ReverseFunction[]} composites 
     * @returns {LinearLayer}
     */
    static sequential(...composites) {

        for (let i = 0; i < composites.length-1; i++) {
            composites[i].composite =  composites[i+1];
        }

        return composites[0];
    }
}